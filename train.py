from torch.utils.data import Dataset, DataLoader, Subset
import torch
from torch.amp import autocast, GradScaler
from model import GPT
from pydantic import BaseModel

from tqdm import tqdm
import time
import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


class MachadoDeAssisDataset(Dataset):
    def __init__(self, ids, block_size):
        super().__init__()

        self.ids = ids
        self.block_size = block_size

    def __len__(self):
        return len(self.ids) - self.block_size

    def __getitem__(self, index):
        chunk = self.ids[index : index + self.block_size + 1]

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)

        return x, y


class TrainConfig(BaseModel):
    data_path: str = "data/machado_de_assis.txt"
    train_size: float = 0.9

    max_vocab_size: int = 8000

    block_size: int = 256
    embedding_dim: int = 384
    n_heads: int = 6
    n_layers: int = 6
    dropout: float = 0.2

    batch_size: int = 64
    max_steps: int = 50_000
    eval_interval: int = 500
    weight_decay: float = 0.1
    learning_rate: float = 3e-4
    patience: int = 5

    seed: int = 42
    compile_model: bool = True


def cycle(dataloader):
    while True:
        for batch in dataloader:
            yield batch

def configure_optimizer(model, lr, weight_decay):
    decay = [p for n, p in model.named_parameters()
             if p.requires_grad and p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters()
                if p.requires_grad and p.dim() < 2]

    return torch.optim.AdamW([
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr, betas=(0.9, 0.95), eps=1e-8)


@torch.no_grad()
def estimate_loss(model, dataloader, device, eval_iters=50):
    model.eval()
    losses = []

    for i, (X, y) in enumerate(dataloader):
        if i >= eval_iters:
            break

        X, y = X.to(device), y.to(device)

        with autocast(device_type=device, dtype=torch.float16):
            _, loss = model(X, y)

        losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)


def train(
    model,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    max_step,
    eval_interval,
    config,
    tokenizer,
    device,
):
    try:
        scaler = GradScaler()

        best_val_loss = float("inf")
        os.makedirs("models/checkpoints", exist_ok=True)

        history = {"train_losses": [], "val_losses": []}

        model.train()
        total_loss = 0
        no_improve = 0

        progress_bar = tqdm(range(max_step), desc="Trainando", leave=False)

        train_iter = cycle(train_loader)

        for step in progress_bar:
            t0 = time.time()

            X_batch, y_batch = next(train_iter)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            with autocast(device_type=device, dtype=torch.float16):
                logits, loss = model(X_batch, y_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

            scheduler.step()

            if step % eval_interval == 0:
                train_loss = total_loss / eval_interval

                val_loss = estimate_loss(model, val_loader, device)
                dt = time.time() - t0

                history["train_losses"].append(train_loss)
                history["val_losses"].append(val_loss)

                print(
                    f"Step {step + 1} | Time: {dt:.2f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
                )

                total_loss = 0

                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "step": step,
                        "best_val_loss": best_val_loss,
                        "config": config}, 
                        "models/checkpoints/best_model.pt")
                else:
                    no_improve += 1
                    if no_improve >= config.patience:
                        print(f"Early stopping no step {step}")
                        break

                model.eval()

                with torch.no_grad():
                    context = torch.zeros((1, 1), dtype=torch.long, device=device)
                    print(
                        tokenizer.decode(
                            model.generate(
                                context, max_new_tokens=100, do_sample=True
                            ).tolist()[0]
                        )
                    )

                model.train()

        return True

    except Exception as e:
        print(str(e))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = TrainConfig()

    torch.manual_seed(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True   
    torch.backends.cudnn.allow_tf32 = True

    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    tokenizer_path = "models/tokenizer.json"
    if os.path.exists(tokenizer_path):
        tokenizer = Tokenizer.from_file(tokenizer_path)
        print("Tokenizer carregado do disco.")
    else:
        trainer = BpeTrainer(
            vocab_size=config.max_vocab_size,
            min_frequency=2,
            special_tokens=["<endoftext>", "<padding>"],
        )
        tokenizer.train([config.data_path], trainer)
        os.makedirs("models", exist_ok=True)
        tokenizer.save(tokenizer_path)
        print("Tokenizer treinado e salvo.")

    with open(config.data_path, "r", encoding="utf-8") as f:
        text = f.read()

    ids = tokenizer.encode(text).ids
    
    dataset = MachadoDeAssisDataset(ids, config.block_size)

    split = int(len(dataset) * config.train_size)
    train_data = Subset(dataset, range(split))
    val_data = Subset(dataset, range(split, len(dataset)))

    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=config.batch_size, num_workers=4, pin_memory=True
    )

    vocab_size = tokenizer.get_vocab_size()

    model = GPT(
        vocab_size=vocab_size,
        n_layers=config.n_layers,
        embedding_dim=config.embedding_dim,
        n_heads=config.n_heads,
        block_size=config.block_size,
        dropout=config.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parâmetros: {n_params / 1e6:.2f}M")

    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
        print("Modelo compilado com torch.compile.")

    optimizer = configure_optimizer(model, config.learning_rate, config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_steps
    )

    train(
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        config.max_steps,
        config.eval_interval,
        config,
        tokenizer,
        device,
    )


if __name__ == "__main__":
    main()

# GPT from Scratch

Implementação de um modelo GPT (Generative Pre-trained Transformer) do zero com PyTorch, treinado sobre obras de **Machado de Assis** para geração de texto em português.

---

## Sobre o Projeto

Este projeto implementa a arquitetura GPT completa. O modelo é treinado sobre um corpus de textos de Machado de Assis e aprende a gerar texto no estilo do autor.

A arquitetura segue de perto o paper *"Attention Is All You Need"* e o GPT-2 da OpenAI, com um decoder transformer composto por blocos de atenção causal e redes feed-forward.

---

## Arquitetura

```
GPT
├── Token Embedding (vocab_size → embedding_dim)
├── Positional Embedding (block_size → embedding_dim)
├── N × DecoderBlock
│   ├── LayerNorm
│   ├── CausalSelfAttention (multi-head)
│   ├── LayerNorm
│   └── MLP (embedding_dim → 4× → embedding_dim, GELU)
├── LayerNorm Final
└── LM Head (embedding_dim → vocab_size)
```

**Hiperparâmetros padrão:**

| Parâmetro       | Valor   |
|-----------------|---------|
| `embedding_dim` | 384     |
| `n_heads`       | 6       |
| `n_layers`      | 6       |
| `block_size`    | 256     |
| `dropout`       | 0.2     |
| `vocab_size`    | 8.000   |

---

## Estrutura do Projeto

```
gpt-from-scratch/
├── model.py                        # Arquitetura do GPT (Attention, Decoder Block, GPT)
├── train.py                        # Loop de treinamento, tokenizer, dataset e configs
├── data/
│   └── machado_de_assis.txt        # Corpus de treinamento
├── models/
│   ├── tokenizer.json              # Tokenizer BPE treinado
│   └── checkpoints/
│       └── best_model.pt           # Melhor checkpoint salvo
└── pyproject.toml                  # Dependências do projeto
```

---

## Instalação

Este projeto usa [uv](https://github.com/astral-sh/uv) para gerenciamento de dependências, com Python 3.13+.

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/gpt-from-scratch.git
cd gpt-from-scratch

# Instale as dependências com uv
uv sync

# Ou com pip tradicional
pip install -e .
```

**Dependências principais:**

- `torch >= 2.10.0`
- `tokenizers >= 0.22.2`
- `pydantic >= 2.12.5`
- `transformers >= 5.1.0`
- `numpy >= 2.4.2`

---

## Como Usar

### Treinamento

```bash
python train.py
```

O script irá:

1. Treinar (ou carregar do disco) um tokenizer BPE sobre o corpus
2. Montar o dataset com janelas deslizantes de `block_size` tokens
3. Instanciar e compilar o modelo GPT via `torch.compile`
5. Salvar o melhor checkpoint em `models/checkpoints/best_model.pt`

A cada `eval_interval` passos, o modelo imprime amostras de texto gerado no terminal.

### Configuração

Todas as opções de treinamento estão centralizadas na classe `TrainConfig` em `train.py`:

```python
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
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    patience: int = 5       

    seed: int = 42
    compile_model: bool = True
```
---

## Recursos de Treinamento

- **AdamW com weight decay seletivo** — aplica weight decay apenas em parâmetros com `dim >= 2` (pesos), não em biases e layer norms
- **Cosine Annealing LR** — taxa de aprendizado decai suavemente ao longo do treinamento
- **Early Stopping** — para o treinamento após `patience` avaliações consecutivas sem melhora na validação
- **torch.compile** — compila o modelo para maior performance (requer PyTorch 2.0+)
- **Checkpointing** — salva automaticamente o melhor modelo com base na perda de validação

---

## Monitoramento

A cada `eval_interval` passos o script exibe:

```
Step 500 | Time: 1.23s | Train Loss: 4.1820 | Val Loss: 4.0934
```

E uma amostra de texto gerado pelo modelo no estado atual.

---

## Formato do Checkpoint

O arquivo `best_model.pt` contém:

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "scaler_state_dict": ...,
    "step": int,
    "best_val_loss": float,
    "config": TrainConfig,
}
```

---

## Referências

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://openai.com/research/better-language-models) — OpenAI, 2019
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Andrej Karpathy

---

## Licença

Distribuído sob a licença MIT.
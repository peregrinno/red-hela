# red-hela

Backend da [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026), inspirado e estendido a partir de [Fiddelis/rinha-backend-2026](https://github.com/Fiddelis/rinha-backend-2026).

## Estratégia

1. **msgspec** — parse JSON tipado, sem overhead de Pydantic.
2. **Starlette** — camada HTTP mínima.
3. **Árvore de decisão** — fast path para folhas com score ≥ 98% de confiança.
4. **IVF 1024 clusters** — busca em células com vetores reordenados (cell-major).
5. **uint8 + rerank float16** — distância barata em lote, refinamento nos top-K.
6. **mmap** — artefatos `.npy` mapeados, baixo RSS no boot.

## Artefatos (`resources/`)

Gerados offline (não vão no build Docker):

| Arquivo | Uso |
|---------|-----|
| `vectors_q8.npy` | distância int8 por lote |
| `vectors_f16.npy` | rerank dos candidatos |
| `labels.npy` | rótulos 0/1 |
| `centroids.npy` | IVF |
| `cluster_offsets.npy` / `cluster_indices.npy` | listas invertidas |
| `tree.npz` | fast path |

## Build offline

```powershell
.\scripts\build_all.ps1
```

Requer `resources/references.json.gz` (repo da rinha).

## Docker

```powershell
docker compose up -d --build
```

## Teste k6

```powershell
cd ..\rinha-de-backend-2026
k6 run test/test.js
```

## Variáveis de tuning

| Variável | Default | Descrição |
|----------|---------|-----------|
| `RED_HELA_IVF_NPROBE` | 2 | células visitadas |
| `RED_HELA_RERANK_K` | 24 | candidatos antes do rerank f16 |
| `RED_HELA_TREE_CONFIDENCE` | 0.98 | limiar da árvore |
| `RED_HELA_CELL_FAST_MARGIN` | 0.0 | atalho por célula pura |

## Stack

Python 3.12, uv, Starlette, msgspec, NumPy, scikit-learn (build), loguru, nginx.

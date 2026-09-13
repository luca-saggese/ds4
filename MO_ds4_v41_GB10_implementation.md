# MO — DS4 V4.1 Flash / DGX Spark GB10
## Piano di implementazione, integrazione PR, test, commit e criteri di accettazione

**Target esclusivo:** NVIDIA DGX Spark / GB10, 128 GB unified memory, CUDA  
**Obiettivo:** portare DeepSeek V4.1 Flash sul backend CUDA/GB10 di `antirez/ds4`, riusando l'infrastruttura SSD streaming già presente, integrando una persistent expert cache corretta e misurabile, e preparando il terreno per tracing, cache policy avanzate e quantizzazione expert-wise.  
**Principio guida:** integrare solo ciò che è utile sul GB10; non importare ottimizzazioni nate per GPU discrete PCIe quando sono neutre o peggiorative su unified memory.

> Questo documento parte dallo stato del repository e delle PR aperte verificato il 13 settembre 2026. Le PR indicate sono ancora aperte: i relativi SHA vanno congelati localmente prima di iniziare.

---

# 0. Decisione architetturale

La baseline da costruire è:

```text
antirez/ds4 main congelato
        │
        ├── CUDA / GB10 backend esistente
        ├── SSD selected-expert streaming esistente
        │
        ├── PR #647
        │     ├── statistiche CUDA streaming
        │     ├── persistent per-(layer,expert) LRU
        │     ├── pooled expert buffers
        │     └── byte-accurate cache budget
        │
        ├── PR #1031
        │     └── IQ2/Q2 prefill usa compact selected-expert cache
        │
        ├── V4.1 CUDA port
        │     ├── model dispatch
        │     ├── CED / CSA2 / indexer
        │     ├── KV V4.1
        │     ├── Engram CUDA + SSD
        │     └── routed experts → existing SSD/cache path
        │
        └── profiling / tracing
              ├── routing stats
              ├── cache hit/miss
              ├── SSD bytes/token
              └── dataset per sensitivity / mixed quant
```

## Non integriamo ora

Non integrare nella baseline:

- PR #589: multi-GPU/MTP/VRAM caching; non serve al target **single GB10**.
- PR #605 intera.
- commit #605 `1c055bb` — pipelined streamed expert upload.
- ottimizzazioni specifiche per PCIe discrete GPU.
- predictor neurale degli expert.
- mixed Q2/Q3/Q4 per singolo expert.
- speculative prefetch.
- multi-Spark.
- MTP/DSpark.
- vision, finché il text path V4.1 non è corretto.
- ottimizzazioni di throughput prima della correttezza numerica.

Il motivo principale per escludere il pipelining #605 è empirico: sul GB10 unified memory è stato misurato un **regresso dell'11–25%** nei workload cold/miss-heavy. Il vantaggio del pipelining nasce dal sovrapporre host-read e PCIe DMA sulle GPU discrete; sul GB10 quel trasferimento non esiste nello stesso modo perché CPU/GPU condividono memoria via NVLink-C2C.

Fonti:
- PR #605: https://github.com/antirez/ds4/pull/605
- discussione/test GB10 dentro #605, commit di valutazione `1242102`
- PR #589: https://github.com/antirez/ds4/pull/589

---

# 1. PR da integrare

## 1.1 PR #647 — INTEGRARE

**PR:** `cuda: implement real per-(layer,expert) LRU for --ssd-streaming-cache-experts`  
URL: https://github.com/antirez/ds4/pull/647

Questa è la base della persistent expert cache CUDA.

Commit da congelare nell'ordine:

```text
062d994  cuda: add DS4_CUDA_STREAM_STATS=1 diagnostic counters...
d84d9f0  cuda: implement a real per-(layer,expert) LRU expert cache...
e59f289  cuda: pool the expert LRU's device buffers...
ab98473  cuda: enforce the expert-cache budget on actual device bytes...
```

La PR introduce:

- cache persistente keyed `(layer, expert)`;
- hit serviti via device-to-device copy verso lo staging buffer già esistente;
- miss letti dal backing model e installati nella cache;
- global LRU;
- buffer pool per evitare `cudaMalloc/cudaFree` ad ogni eviction;
- `DS4_CUDA_STREAM_STATS=1`;
- hit/miss counters;
- bytes from file / bytes from cache;
- budget realmente contabilizzato in **byte device effettivi**;
- conteggio anche dei buffer parcheggiati nelle free-list;
- protezione contro un bug reale osservato su GB10, dove un budget nominale da 100 GB arrivava a circa 121 GB reali.

Dati riportati dalla PR su GPU GB10-class:

```text
100 GB cache:
1.03 -> 2.96 tok/s
~81% hit rate

sessione lunga:
~96-98% hit rate

20 GB:
1.62 -> 4.41 tok/s
~51.5% hit rate
```

Per il nostro progetto è la PR prioritaria.

---

## 1.2 PR #1031 — INTEGRARE

**PR:** `CUDA: use selected expert cache for IQ2 SSD prefill`  
URL: https://github.com/antirez/ds4/pull/1031

Commit:

```text
9b8d8fd  cuda: use selected expert cache for IQ2 prefill
```

La patch corregge il batch-prefill IQ2/Q2: se il selected-expert staging buffer è già stato costruito, il kernel MMQ deve usare:

```text
compact selected experts
+
remapped selected-slot tensor
```

e non risolvere nuovamente le full MoE expert table.

A/B riportato su DGX Spark GB10:

```text
prefill:
1.09 -> 2.59 tok/s

CUDA file cache:
78.62 GiB -> 6.06 GiB

output:
uguale nel test
```

Questa PR non è la persistent LRU: lavora più a valle, sul buffer compatto usato dal compute.

È complementare a #647.

---

## 1.3 PR #605 — NON CHERRY-PICKARE INTERA

**PR:** `CUDA: resident expert cache and faster selected-expert uploads for --ssd-streaming decode`  
URL: https://github.com/antirez/ds4/pull/605

Commit principali:

```text
10ba298  restore resident expert cache
9167d12  skip per-load VRAM query once cache settled
1c055bb  pipeline streamed expert upload path
5930a3f  plain pageable uploads when no O_DIRECT
```

### Non importare `10ba298` come blocco

La resident cache di #605 sovrappone la funzione di #647 e introdurrebbe due implementazioni concorrenti.

Per il GB10 usiamo **#647 come autorità unica della persistent cache**.

### Idee da valutare manualmente dopo #647

Da #605 possono essere riprese, solo se mancanti e dopo profiling:

- prefill può **leggere la cache**;
- durante prefill si può **riempire capacità libera senza fare eviction** degli expert warm;
- hotlist/preload hooks;
- eventuale eliminazione di query `cudaMemGetInfo` nel path caldo se il profiler dimostra che esistono ancora dopo #647.

Queste logiche vanno portate come commit nostri, non cherry-pickando la resident cache di #605.

### Escludere `1c055bb`

Non importare:

```text
1c055bb  pipeline the streamed expert upload path
```

Sul GB10 il port di questa logica ha mostrato regressioni:

```text
cold turn:
~11-22% più lento

miss-heavy:
~25% più lento
```

La spiegazione è coerente con l'architettura: il pipelining nasconde PCIe DMA latency su GPU discrete; il DGX Spark usa unified memory e NVLink-C2C.

### `5930a3f`

Non serve come obiettivo iniziale.

Su unified-memory il path host-resident/plain-copy può essere neutro o naturale, ma non va introdotto finché non abbiamo misure specifiche V4.1 + SSD reale.

---

## 1.4 PR #589 — FUORI SCOPE

URL: https://github.com/antirez/ds4/pull/589

Contiene multi-GPU SSD streaming, MTP e VRAM caching.

Non serve al primo target:

```text
1 x DGX Spark
1 x GB10
```

Non integrarla.

---

# 2. Regola fondamentale: nessun merge “bulk”

Non usare:

```bash
git merge refs/pull/647/head
```

come unico commit.

Non usare:

```bash
git merge refs/pull/605/head
```

Non squashare PR eterogenee.

Ogni cambiamento deve restare:

- isolabile;
- testabile;
- revertibile;
- bisectabile.

La storia Git deve permettere:

```bash
git bisect
```

tra:

- baseline;
- statistiche;
- persistent LRU;
- pool;
- byte accounting;
- #1031;
- V4.1 scaffolding;
- singole primitive V4.1;
- Engram;
- streaming V4.1.

---

# 3. Clone e freeze iniziale

## 3.1 Clone pulito

```bash
mkdir -p ~/src
cd ~/src

git clone https://github.com/antirez/ds4.git ds4-mo
cd ds4-mo
```

## 3.2 Verifica remote

```bash
git remote -v
```

Deve risultare almeno:

```text
origin  https://github.com/antirez/ds4.git
```

## 3.3 Aggiornamento e freeze

```bash
git fetch origin --tags
git checkout main
git pull --ff-only origin main
```

Salvare immediatamente:

```bash
BASE_SHA=$(git rev-parse HEAD)
echo "$BASE_SHA"
```

Creare tag locale:

```bash
git tag -a mo-gb10-base-2026-09-13 \
    -m "MO GB10 frozen upstream baseline 2026-09-13"
```

Creare branch:

```bash
git switch -c mo/gb10-v41
```

## 3.4 Manifest del freeze

Creare:

```text
MO_FREEZE.md
```

contenente:

```text
Date
Upstream SHA
Kernel
Ubuntu version
CUDA version
Driver version
GPU name
GPU architecture
RAM
SSD model
SSD filesystem
Compiler
NVCC
Model GGUF filename
Model SHA256
```

Comandi utili:

```bash
git rev-parse HEAD
git status --short
uname -a
cat /etc/os-release
nvcc --version
nvidia-smi
lspci
lsblk -o NAME,MODEL,SIZE,FSTYPE,MOUNTPOINT
df -hT
```

Per il modello:

```bash
sha256sum /path/to/model.gguf
```

### Commit MO-000

```bash
git add MO_FREEZE.md
git commit -m "mo: freeze GB10 baseline and environment manifest"
```

Questo commit non deve contenere codice.

---

# 4. Fetch delle PR senza contaminarle

Creare ref locali:

```bash
git fetch origin pull/647/head:refs/heads/pr/647
git fetch origin pull/1031/head:refs/heads/pr/1031
git fetch origin pull/605/head:refs/heads/pr/605
git fetch origin pull/589/head:refs/heads/pr/589
```

Verificare:

```bash
git log --oneline --decorate pr/647 -10
git log --oneline --decorate pr/1031 -5
git log --oneline --decorate pr/605 -10
```

Salvare gli SHA realmente presenti.

Non assumere che una PR aperta mantenga per sempre lo stesso HEAD.

Dopo il freeze, il nostro branch usa gli SHA congelati anche se GitHub cambia.

---

# 5. Baseline GB10 prima delle integrazioni

Prima di cherry-pickare qualsiasi PR dobbiamo dimostrare che la baseline compila e gira.

## 5.1 Build

```bash
make clean
make -j"$(nproc)" cuda-spark
```

Gate:

```text
PASS = build completa senza errori
FAIL = stop
```

## 5.2 Test upstream

Eseguire almeno:

```bash
./ds4_test
```

e dove disponibile:

```bash
make test-cuda-q8-scratch
```

Annotare eventuali failure già presenti nella baseline.

Non correggere failure preesistenti nello stesso branch.

Creare:

```text
results/baseline-tests.txt
```

## 5.3 Smoke non-streaming

Con un modello DeepSeek V4 già supportato e che entra:

```bash
./ds4 \
  -m /path/to/reference-v4.gguf \
  --cuda \
  --ctx 4096 \
  --temp 0 \
  --tokens 64 \
  -p "Reply with exactly one short sentence about the Moon."
```

Salvare:

- stdout;
- stderr;
- output token;
- tok/s.

## 5.4 Smoke SSD streaming

```bash
./ds4 \
  -m /path/to/reference-v4.gguf \
  --cuda \
  --ssd-streaming \
  --ctx 4096 \
  --temp 0 \
  --tokens 64 \
  -p "Reply with exactly one short sentence about the Moon."
```

Questo diventa il reference prima della cache.

### Commit MO-001

Non modificare codice.

Committare solo risultati/config:

```bash
git add results/
git commit -m "mo: record GB10 upstream baseline tests"
```

---

# 6. Integrazione #647 in quattro commit distinti

Non cherry-pickare la PR come singolo blocco.

## MO-010 — osservabilità

Cherry-pick:

```bash
git cherry-pick 062d994
```

Obiettivo:

```text
DS4_CUDA_STREAM_STATS=1
```

deve mostrare:

- fetch_calls;
- expert_fetches;
- cache_hits;
- cache_misses;
- bytes_from_file;
- bytes_from_cache.

In questa fase, prima della LRU, ci aspettiamo strutturalmente:

```text
hits = 0
bytes_from_cache = 0
```

### Test

```bash
DS4_CUDA_STREAM_STATS=1 ./ds4 \
  -m /path/to/reference-v4.gguf \
  --cuda \
  --ssd-streaming \
  --ctx 4096 \
  --temp 0 \
  --tokens 64 \
  -p "Explain in one paragraph why the sky is blue."
```

Gate:

- build PASS;
- output greedy identico alla baseline;
- counters visibili;
- hit = 0 coerente.

Commit è già quello cherry-pickato; non aggiungere altre modifiche.

---

## MO-011 — persistent per-(layer,expert) LRU

Cherry-pick:

```bash
git cherry-pick d84d9f0
```

Ora deve esistere una vera persistent cache.

### A/B obbligatorio

Arm A:

```text
cache disabled
```

Arm B:

```text
cache enabled
```

Usare stesso:

- modello;
- prompt;
- `--temp 0`;
- context;
- token count;
- processo quando possibile;
- ordine alternato nelle repliche.

Gate correttezza:

```text
generated token bytes A == generated token bytes B
```

Gate cache:

```text
hits > 0 dopo warm-up
bytes_from_cache > 0
```

Non imporre ancora un target di tok/s.

La prima accettazione è **funzionale**, non prestazionale.

---

## MO-012 — pooled buffers

Cherry-pick:

```bash
git cherry-pick e59f289
```

Obiettivo:

evitare:

```text
cudaMalloc
cudaFree
```

sul path caldo di ogni miss/eviction.

### Test thrash

Usare deliberatamente una cache piccola per creare miss continui.

Esempio concettuale:

```bash
--ssd-streaming-cache-experts 8GB
```

Misurare:

- stabilità;
- tok/s;
- OOM;
- allocation errors;
- counters.

Gate:

- nessun crash;
- output invariato;
- small-cache non deve collassare per alloc/free churn.

---

## MO-013 — byte-accurate budget

Cherry-pick:

```bash
git cherry-pick ab98473
```

Questo commit è **obbligatorio per GB10**.

È stato aggiunto dopo aver osservato sul GB10:

```text
budget nominale 100 GB
uso reale circa 121 GB
```

Il nuovo accounting include:

- bytes degli entry validi;
- bytes nelle pool free-list;
- persistent pinned staging;
- actual device total.

### Test budget

Primi valori:

```text
48 GB
64 GB
70 GB
80 GB
```

Non partire da 100 GB.

Per ciascuno:

- avvia;
- warm-up;
- almeno 200 decode token;
- registra peak memory;
- verifica che il cache footprint non superi il budget pianificato.

Gate:

```text
actual expert-cache device bytes <= configured budget + piccolo overhead esplicitamente contabilizzato
```

Se il sistema entra in memory pressure o swap/availability thrash, il test fallisce anche se non crasha.

---

# 7. Integrazione #1031

## MO-020 — compact selected expert prefill

Cherry-pick:

```bash
git cherry-pick 9b8d8fd
```

Questa patch riguarda IQ2/Q2 batch-prefill.

### Build test

```bash
make clean
make -j"$(nproc)" cuda-spark
make test-cuda-q8-scratch
```

### A/B prefill

Usare un GGUF compatibile con IQ2/Q2 e SSD streaming.

Misurare:

- prefill tok/s;
- file cache;
- RSS/unified memory;
- output finale;
- selected expert counters.

Criterio principale:

```text
nessun accesso sistematico alle full MoE expert tables
quando il selected staging cache è valido
```

Criterio correttezza:

```text
output greedy identico
```

Criterio memoria:

```text
file cache sostanzialmente inferiore alla baseline
```

Non imporre il numero esatto 78.62 → 6.06 GiB: quello è il benchmark della PR, non il nostro gate.

---

# 8. Checkpoint di integrazione cache

A questo punto creare tag:

```bash
git tag -a mo-gb10-cache-v1 \
  -m "GB10 persistent CUDA expert cache + compact IQ2/Q2 prefill"
```

Branch:

```text
mo/gb10-v41
```

deve ora contenere solo:

```text
upstream frozen
+
#647
+
#1031
+
test artifacts
```

Nessun codice V4.1 CUDA ancora.

Questo è il punto da cui fare sempre rollback se il port V4.1 introduce regressioni.

---

# 9. Cosa prendere da #605 e cosa no

## Non cherry-pickare #605

Non fare:

```bash
git cherry-pick 10ba298
```

perché duplicherebbe la resident cache già introdotta da #647.

## Valutare manualmente solo due policy

Dopo aver verificato #647:

### Policy A — prefill non deve distruggere il working set warm

Desiderato:

```text
prefill:
- HIT consentiti
- install in free capacity consentito
- eviction di warm decode experts vietata
```

### Policy B — hotlist / seed hooks

Se le API:

```text
seed_selected
seed_experts
```

restano no-op o non completamente wired dopo #647, implementarle come commit separato.

Non importare altro.

---

# 10. MO-030 — GB10 prefill cache-protection

Solo se i test mostrano che il prefill lungo degrada il decode warm.

Implementare:

```text
prepare_selected_batch():
    consult resident cache
    use hit
    if free_capacity:
        install miss
    else:
        do not evict warm resident entries
```

Commit:

```bash
git commit -m "cuda-gb10: preserve warm expert cache across batch prefill"
```

### Test

1. warm decode;
2. registra cache hit;
3. esegui prompt lungo;
4. decode successivo;
5. confronta hit rate prima/dopo.

Gate:

```text
prefill non deve azzerare il decode working set
```

---

# 11. MO-031 — expert hotlist / seed hooks

Solo se non già realmente funzionanti dopo #647.

Implementare:

```text
seed_selected()
seed_experts()
```

sulla stessa cache persistente.

Commit:

```bash
git commit -m "cuda-gb10: wire expert cache seed and hotlist hooks"
```

### Gate

Un set di expert seedato deve risultare:

```text
resident before first decode use
```

e produrre:

```text
cache hit on first routed access
```

---

# 12. Non implementare ancora predictor

A questo punto non aggiungere:

- MLP;
- tiny encoder;
- cross-layer neural predictor;
- speculative expert loads.

Prima dobbiamo misurare la LRU semplice.

Se con V4.1 otteniamo:

```text
>90% warm hit rate
```

il predictor potrebbe avere poco ROI.

La decisione viene presa solo dopo il tracer.

---

# 13. Port V4.1 CUDA — strategia

Il port V4.1 va fatto **dopo** aver stabilizzato il sottosistema cache su un modello CUDA già supportato.

Motivo:

se cache e V4.1 vengono modificati insieme non possiamo distinguere:

```text
errore graph
vs
errore streaming
vs
errore cache
vs
errore quantizzazione
```

La reference semantica è l'implementazione Metal V4.1 già presente in `main`.

---

# 14. MO-100 — V4.1 model dispatch, nessun kernel nuovo

Primo commit V4.1 CUDA:

- riconoscimento architecture;
- parsing metadata;
- tensor lookup;
- model shape validation;
- nessuna ottimizzazione.

Obiettivo:

```text
CUDA build riconosce V4.1
carica metadata
fallisce in modo esplicito alla prima primitive non implementata
```

Non deve esistere un silent fallback.

Commit:

```bash
git commit -m "cuda-v41: add model dispatch and tensor metadata plumbing"
```

### Gate

- V4 precedente non cambia;
- V4.1 viene identificato correttamente;
- mismatch tensor produce errore chiaro;
- nessun crash.

---

# 15. MO-101 — primitive V4.1 CUDA: port incrementale

Non creare un mega-commit.

Portare gruppi di primitive omogenee.

Ordine raccomandato:

```text
A. quant/dequant/layout helpers
B. RoPE / positional path
C. CSA2/indexer support
D. attention output path
E. CED-specific state/dataflow
F. KV operations
G. Engram projection helpers
```

Ogni gruppo deve avere:

```text
commit
+
unit/scratch test
+
reference comparison
```

Esempio commit:

```text
cuda-v41: port V4.1 quantization helpers to GB10
cuda-v41: port V4.1 rope and positional kernels
cuda-v41: add CSA2 indexer kernels
cuda-v41: add CED attention dataflow
cuda-v41: implement V4.1 KV operations
```

Non usare un commit:

```text
"add V4.1 CUDA support"
```

da migliaia di righe.

---

# 16. Reference numerica per il port

Per ogni primitive o blocco salvare golden output.

Reference preferita:

```text
V4.1 Metal implementation
```

su identici:

- pesi;
- input token;
- layer;
- tensor shape;
- seed/deterministic path.

Se non è disponibile localmente, conservare golden dump generati una volta da un host Metal/reference.

Per ogni blocco confrontare almeno:

```text
max_abs_error
mean_abs_error
relative_error
top-k router IDs quando applicabile
```

Per il modello completo:

```text
greedy token sequence
```

è il gate finale.

---

# 17. MO-110 — V4.1 router senza SSD optimization

Prima far funzionare il router correttamente.

Output da loggare:

```text
layer
token
top-6 expert IDs
router score top-6
```

Commit:

```bash
git commit -m "cuda-v41: implement routed expert selection and trace hooks"
```

A questo punto non usare predictor.

Gate:

```text
top-6 IDs coincidono con reference per i token di test
```

o, se la precisione interna differisce legittimamente:

```text
differenze spiegate e output finale validato
```

---

# 18. MO-111 — V4.1 routed expert compute residente/diagnostico

Prima di SSD streaming, verificare il compute expert su un piccolo test/synthetic tensor se l'intero modello non entra.

Il test deve coprire:

```text
FP4 expert
top-6 selection
gate/up/down
selected-slot remap
reduction
```

Il backend CUDA ha già supporto MXFP4/Blackwell: non riscrivere i kernel FP4 se quelli esistenti possono essere riusati.

Commit:

```bash
git commit -m "cuda-v41: wire routed FP4 experts into existing GB10 MoE path"
```

---

# 19. MO-120 — V4.1 Engram CUDA

Engram è il secondo blocco realmente nuovo.

Obiettivo:

```text
token/ngram
    ↓
hash/index
    ↓
SSD-backed table lookup
    ↓
staging
    ↓
CUDA projection/add
```

Il comportamento matematico va copiato dalla reference Metal, non reinventato.

Separare almeno:

### MO-120A

```text
Engram index/hash parity
```

Commit:

```bash
git commit -m "cuda-v41: implement Engram index and hash parity"
```

### MO-120B

```text
Engram SSD row reader
```

Commit:

```bash
git commit -m "cuda-v41: add SSD-backed Engram row access for GB10"
```

### MO-120C

```text
Engram CUDA projection/add
```

Commit:

```bash
git commit -m "cuda-v41: integrate Engram projection into CUDA graph"
```

---

# 20. Engram I/O policy GB10

Non usare lo stesso criterio degli expert.

Expert:

```text
letture grandi
~MiB per expert
```

Engram:

```text
lookup piccoli e sparsi
```

Per il primo implementation:

- priorità correttezza;
- nessun caching sofisticato;
- nessun prefetch speculativo;
- nessun predictor.

Misurare prima.

---

# 21. MO-130 — collega V4.1 all'existing expert SSD streaming

A questo punto il V4.1 router deve alimentare il sottosistema che abbiamo già validato con #647.

Flusso:

```text
V4.1 router top-6
        ↓
persistent LRU peek(layer,expert)
       / \
     HIT MISS
      |    |
      |   SSD
      |    |
      +----+
        ↓
compact selected staging
        ↓
FP4 routed expert compute
```

Commit:

```bash
git commit -m "cuda-v41: connect routed experts to persistent SSD streaming cache"
```

### Gate

- cache disabled → output corretto;
- cache enabled → stesso output;
- `hits > 0` dopo warm-up;
- cache budget rispettato;
- cache miss legge soltanto selected expert;
- nessun full-table fetch inatteso.

---

# 22. MO-131 — V4.1 prefill integration

Il prefill deve usare:

- selected expert staging;
- existing compact execution;
- persistent cache quando opportuno;
- prefill no-evict policy se implementata.

Commit:

```bash
git commit -m "cuda-v41: integrate selected-expert SSD path into batch prefill"
```

Non assumere che #1031 copra automaticamente FP4 V4.1: #1031 modifica esplicitamente il path IQ2/Q2 MMQ.

Il V4.1 FP4 path va verificato separatamente.

---

# 23. Primo end-to-end milestone

Tag:

```bash
git tag -a mo-v41-gb10-first-token \
  -m "DeepSeek V4.1 first correct greedy token on DGX Spark"
```

Criterio per creare il tag:

```text
V4.1 load PASS
Engram PASS
router PASS
selected FP4 expert compute PASS
SSD expert streaming PASS
greedy output matches reference for smoke prompt
no illegal memory access
no OOM
```

La velocità non conta ancora.

---

# 24. Test matrix minima

## T0 — build

```text
make cuda-spark
```

PASS obbligatorio ad ogni commit.

## T1 — upstream regression

Modello DeepSeek V4 già supportato.

Obiettivo:

```text
nessuna regressione funzionale
```

## T2 — cache OFF vs ON

```text
temp=0
same prompt
same output bytes
```

## T3 — cold cache

Nuovo processo / cache vuota.

## T4 — warm cache

Seconda/terza generazione nello stesso processo.

## T5 — small cache thrash

```text
8/16 GB
```

## T6 — practical cache

```text
48/64/70/80 GB
```

## T7 — long prompt

Verifica che il prefill non distrugga la cache decode.

## T8 — V4.1 Engram

Prompt con sufficiente lunghezza per esercitare gli Engram layer.

## T9 — V4.1 SSD miss-heavy

Cache volutamente piccola.

## T10 — V4.1 warm locality

Sessione lunga/multi-turn.

---

# 25. Metriche da salvare in ogni run

Formato CSV o JSON Lines.

Per run:

```text
git_sha
model_sha256
prompt_id
ctx
generated_tokens
cache_budget_gib
prefill_tok_s
decode_tok_s
wall_time
rss
gpu/unified memory peak
cache_hits
cache_misses
cache_hit_rate
bytes_from_file
bytes_from_cache
expert_cache_counted_bytes
expert_cache_parked_bytes
expert_cache_total_bytes
ssd_read_bytes
ssd_read_bw
output_sha256
```

Per V4.1 aggiungere:

```text
engram_read_bytes
engram_lookup_count
engram_latency
router_trace_file
```

---

# 26. Test deterministico standard

Usare sempre almeno un test con:

```text
temperature = 0
fixed prompt
fixed token cap
```

Salvare:

```bash
sha256sum output.txt
```

Il test A/B passa solo se:

```text
output_a.sha256 == output_b.sha256
```

per le modifiche che devono essere byte-transparent:

- cache;
- staging;
- allocator;
- budget accounting;
- I/O policy.

Per quantizzazioni future questo gate cambierà.

---

# 27. Profiling dopo il primo V4.1 corretto

Prima di mixed quant/predictor, misurare:

```text
cache budget -> hit rate
```

per:

```text
16
32
48
64
70
80 GiB
```

con almeno:

```text
general
code
math
reasoning
Italian
English
Chinese
long-context
```

Per ogni dominio:

```text
cold hit rate
warm hit rate
bytes/token
misses/token
layers with >=1 miss/token
decode tok/s
```

Decisione successiva basata sui dati.

---

# 28. Tracer V4.1

Estendere la strumentazione solo dopo il runtime corretto.

Commit separati:

## MO-200

```bash
git commit -m "trace: record V4.1 routed expert ids and router scores"
```

Per event:

```text
session_id
token_index
layer
expert_id[6]
score[6]
```

## MO-201

```bash
git commit -m "trace: add per-domain expert frequency aggregates"
```

## MO-202

```bash
git commit -m "trace: add temporal and cross-layer expert transitions"
```

## MO-203

```bash
git commit -m "trace: add offline cache-policy simulator"
```

Non salvare 384 logits per layer/token se non servono.

---

# 29. Quando iniziare sensitivity Q2/Q3/Q4

Solo quando abbiamo:

```text
V4.1 correct
+
persistent cache correct
+
trace dataset
+
stable benchmark harness
```

Non prima.

La futura sensitivity analysis dovrà separare:

```text
frequency
```

da:

```text
quantization sensitivity
```

e potrà produrre:

```text
expert X -> Q4
expert Y -> Q3
expert Z -> Q2
```

Ma non fa parte della prima integration milestone.

---

# 30. Quando iniziare predictor/prefetch

Solo dopo aver misurato la LRU.

Regola:

```text
se warm hit rate >= 90-95%
    predictor = bassa priorità
else
    valutare transition predictor
```

Prima baseline:

```text
transition table
```

non rete neurale.

Solo se le transition table non bastano:

```text
small neural predictor
```

---

# 31. Commit map completa

Sequenza raccomandata:

```text
MO-000  mo: freeze GB10 baseline and environment manifest
MO-001  mo: record GB10 upstream baseline tests

MO-010  #647 stats       [062d994]
MO-011  #647 LRU         [d84d9f0]
MO-012  #647 pool        [e59f289]
MO-013  #647 byte budget [ab98473]

MO-020  #1031 selected IQ2/Q2 prefill [9b8d8fd]

MO-030  cuda-gb10: preserve warm expert cache across batch prefill
        [solo se necessario]

MO-031  cuda-gb10: wire expert cache seed and hotlist hooks
        [solo se necessario]

MO-100  cuda-v41: add model dispatch and tensor metadata plumbing
MO-101A cuda-v41: port quant/layout helpers
MO-101B cuda-v41: port rope/position path
MO-101C cuda-v41: add CSA2/indexer
MO-101D cuda-v41: add CED dataflow
MO-101E cuda-v41: add V4.1 KV operations

MO-110  cuda-v41: implement router and trace hooks
MO-111  cuda-v41: wire routed FP4 experts into GB10 MoE path

MO-120A cuda-v41: implement Engram index/hash parity
MO-120B cuda-v41: add SSD-backed Engram access
MO-120C cuda-v41: integrate Engram CUDA projection

MO-130  cuda-v41: connect routed experts to persistent SSD cache
MO-131  cuda-v41: integrate selected-expert batch prefill

TAG     mo-v41-gb10-first-token

MO-200  trace: routed IDs and scores
MO-201  trace: domain aggregates
MO-202  trace: transition statistics
MO-203  trace: offline cache simulator
```

Ogni commit:

```text
1 concetto
1 test
1 rollback point
```

---

# 32. Branch policy

Usare:

```text
mo/gb10-v41
```

come integration branch.

Creare worktree separati:

```bash
git worktree add ../ds4-base mo-gb10-base-2026-09-13
git worktree add ../ds4-mo mo/gb10-v41
```

Per esperimenti rischiosi:

```text
exp/605-prefill-policy
exp/v41-engram
exp/v41-router
exp/cache-trace
```

Mai sviluppare un esperimento direttamente sopra il branch stabile.

---

# 33. Rebase/upstream policy

Durante il port:

**non inseguire `main` ad ogni commit upstream.**

Il freeze serve a evitare una moving target.

Integrare upstream solo a milestone:

```text
M0 cache stable
M1 first V4.1 token
M2 V4.1 text stable
M3 tracer stable
```

Procedura:

```bash
git fetch origin
git log --oneline HEAD..origin/main
```

Valutare manualmente.

Mai:

```bash
git pull
```

automatico sul branch MO.

---

# 34. Criteri di STOP

Fermare l'integrazione e non procedere al commit successivo se:

- build CUDA fallisce;
- V4 baseline cambia output senza spiegazione;
- cache ON/OFF cambia output a `temp=0`;
- actual cache bytes superano significativamente il budget;
- compare illegal memory access;
- CUDA context viene poisoned;
- SSD streaming legge full MoE table quando selected cache è valida;
- prefill distrugge sistematicamente il warm working set;
- GB10 entra in memory pressure/swap;
- una “ottimizzazione” migliora GPU discrete ma peggiora GB10.

Non accumulare fix sopra una failure non capita.

---

# 35. Criteri di successo della prima fase

La fase di integrazione è conclusa quando abbiamo:

```text
[ ] upstream frozen e riproducibile
[ ] build cuda-spark stabile
[ ] #647 integrata commit-per-commit
[ ] #1031 integrata
[ ] budget expert cache byte-correct
[ ] cache hits reali sul GB10
[ ] output cache ON/OFF identico
[ ] V4 baseline senza regressioni
[ ] V4.1 text graph su CUDA
[ ] V4.1 router corretto
[ ] V4.1 native FP4 expert compute
[ ] V4.1 Engram su SSD
[ ] V4.1 expert SSD streaming
[ ] V4.1 persistent cache
[ ] V4.1 first greedy output corretto
[ ] cache stats e bytes/token disponibili
```

Non serve ancora:

```text
[ ] mixed quant per expert
[ ] sensitivity profiler
[ ] neural predictor
[ ] speculative prefetch
[ ] vision
[ ] multi-GPU
```

---

# 36. Configurazione GB10 iniziale prudente

Non partire con una cache da 100 GB.

La #647 ha documentato che un bug precedente poteva portare 100 GB nominali a ~121 GB reali; il fix `ab98473` risolve l'accounting, ma il sistema ha comunque bisogno di headroom.

Per i primi test:

```text
48 GiB
64 GiB
70 GiB
```

Poi:

```text
80 GiB
```

solo dopo aver misurato:

- runtime buffers;
- context;
- KV;
- Engram staging;
- V4.1 dense residency;
- peak durante prefill.

Il budget ottimo non è necessariamente il massimo possibile.

---

# 37. GB10-specific: cosa evitare

## Evitare 1 — PCIe-style upload pipelining

Non integrare:

```text
#605 1c055bb
```

senza nuovi dati che lo contraddicano.

## Evitare 2 — cache enorme senza headroom

Più cache non implica automaticamente più throughput.

Memory pressure può peggiorare:

- allocation;
- page behavior;
- staging;
- context creation.

## Evitare 3 — query runtime costose per layer/token

Se una funzione come:

```text
cudaMemGetInfo
```

compare nel path caldo, profilarla.

Ma non portare automaticamente `9167d12`: verificare prima se #647 ha lo stesso problema.

## Evitare 4 — duplicare due cache implementation

Scegliere una sola autorità:

```text
#647
```

Non sovrapporre #605 resident cache.

## Evitare 5 — ottimizzare decode prima di misurare bandwidth

Su GB10 ds4 ha già osservato che alcuni decode residenti arrivano vicino all'85–90% della bandwidth fisicamente misurata. Le ottimizzazioni utili per il nostro caso devono quindi colpire soprattutto:

```text
SSD misses
cache locality
unnecessary copies
routing/I/O stalls
```

non micro-ottimizzazioni casuali del kernel.

---

# 38. Output finale atteso del progetto MO

Dopo la prima fase il runtime dovrebbe assomigliare a:

```text
                    DGX Spark / GB10
                    128 GB unified

          ┌──────────────────────────┐
          │ V4.1 dense / CED / CSA2 │
          │ router                   │
          │ shared compute           │
          │ KV/runtime               │
          │                          │
          │ persistent expert LRU    │
          │  (layer,expert)          │
          │                          │
          │ selected staging         │
          └────────────┬─────────────┘
                       │ miss
                       ▼
                  NVMe backing
             ┌────────────────────┐
             │ routed experts     │
             │ Engram tables      │
             └────────────────────┘
```

Con metriche osservabili:

```text
expert cache hit rate
expert cache miss rate
bytes from SSD
bytes from cache
actual cache bytes
parked pool bytes
prefill tok/s
decode tok/s
Engram I/O
```

Questo è il punto corretto da cui iniziare la seconda fase:

```text
frequency
+
sensitivity
+
Q2/Q3/Q4 per expert
+
eventuale predictor
```

---

# 39. Fonti usate per le scelte d'integrazione

## Repository

- ds4 main  
  https://github.com/antirez/ds4

- Models documentation  
  https://github.com/antirez/ds4/blob/main/docs/MODELS.md

## PR principali

- #647 — persistent per-(layer,expert) CUDA LRU  
  https://github.com/antirez/ds4/pull/647

- #1031 — selected expert cache nel prefill IQ2/Q2  
  https://github.com/antirez/ds4/pull/1031

- #605 — resident expert cache + upload experiments  
  https://github.com/antirez/ds4/pull/605

- #589 — multi-GPU SSD/MTP/cache  
  https://github.com/antirez/ds4/pull/589

## Evidenza GB10 importante

Nella discussione di #605 è documentato un test indipendente del transport pipelining su GB10 unified memory:

```text
11–22% regression cold
~25% regression miss-heavy
```

Il contributor e l'autore della PR concordano sul fatto che il pipelining è pensato per nascondere PCIe DMA latency e può essere puro overhead su unified memory.

## Altra evidenza GB10

Issue ds4:

- GB10 decode bandwidth analysis  
  https://github.com/antirez/ds4/issues/773

Questa evidenza rafforza la scelta di concentrare il lavoro su:

```text
I/O
expert cache
miss bytes
working set
```

piuttosto che su micro-ottimizzazioni non misurate del decode resident.

---

# 40. Regola finale del MO

La sequenza non va invertita:

```text
CORRETTEZZA
    ↓
OSSERVABILITÀ
    ↓
CACHE CORRETTA
    ↓
V4.1 CUDA
    ↓
ENGRAM + SSD
    ↓
MISURE REALI
    ↓
QUANTIZZAZIONE
    ↓
PREDIZIONE
```

Non sviluppare una cache predittiva per compensare un persistent cache path non ancora corretto.

Non sviluppare mixed quantization prima di avere un golden trace.

Non ottimizzare una primitive V4.1 prima che il graph produca gli stessi risultati del reference.

Non importare un'ottimizzazione nata per PCIe discrete GPU senza un A/B sul GB10.

Ogni miglioramento deve superare tre gate:

```text
1. correctness
2. memory/budget correctness
3. GB10 A/B performance
```

Solo dopo passa nel branch MO stabile.

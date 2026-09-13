# M1 — DeepSeek V4.1 Flash Q2 cold SSD streaming su DGX Spark / GB10
## Specifica di implementazione per ottenere il primo end-to-end CUDA funzionante, anche a ~0,5 tok/s

**Target esclusivo:** NVIDIA DGX Spark / GB10, 128 GB unified memory, CUDA  
**Modello:** `DeepSeek-V4.1-Flash-Q2.gguf` prodotto/distribuito da DwarfStar (`ds41f-q2`)  
**Milestone:** **M1 = primo V4.1 Q2 text-only funzionante sullo Spark con routed expert letti da SSD on demand, Engram su SSD, nessuna persistent expert cache richiesta, nessun predictor, nessuna mixed precision dinamica.**

> M1 è deliberatamente una milestone di **correttezza e completezza del cold path**, non di performance.  
> Se il modello genera correttamente a circa **0,5 token/s**, senza deadlock, OOM, expert mancanti o errori Engram, M1 è raggiunta.

---

# 0. Definizione formale di M1

M1 è completata quando, su **un singolo DGX Spark / GB10**, il seguente percorso funziona end-to-end:

```text
DeepSeek V4.1 Flash Q2 GGUF
          │
          ▼
      CUDA / GB10
          │
          ├── V4.1 CED / CSA2 / indexer
          ├── V4.1 KV
          ├── router top-6
          │
          ├── Engram lookup da SSD
          │
          └── routed expert Q2 da SSD
                    │
                    ▼
           selected-expert staging
                    │
                    ▼
             CUDA Q2 MoE compute
                    │
                    ▼
                 logits
                    │
                    ▼
              greedy token
```

Il run deve essere possibile con:

```text
persistent expert cache = OFF / non necessaria
expert preload          = OFF
predictive prefetch      = OFF
hot expert seeding       = OFF
mixed per-expert bits    = OFF
MTP / DSpark             = OFF
vision                   = OFF
multi-GPU                = OFF
```

Il requisito è:

```text
SSD miss
   ↓
leggi l'expert necessario
   ↓
staging
   ↓
compute
   ↓
continua
```

anche se questo accade praticamente a ogni layer e a ogni token.

---

# 1. Perché M1 deve essere "cold first"

Il progetto completo vuole arrivare a:

```text
SSD streaming
+
persistent expert cache
+
frequency / hotness
+
sensitivity
+
Q2/Q3/Q4 per expert
+
eventuale predictor
```

Ma nessuna di queste ottimizzazioni deve mascherare un errore del port V4.1.

Per M1 vogliamo poter dire:

> **anche senza alcun riuso degli expert, il graph CUDA V4.1, Engram e il Q2 SSD path sono corretti.**

Se M1 funziona, ogni ottimizzazione successiva può essere misurata contro una baseline estremamente semplice.

Se M1 non funziona, non dobbiamo investigare contemporaneamente:

- cache eviction;
- cache hit errati;
- hotness;
- predictor;
- size class;
- mixed quant;
- preload.

---

# 2. Stato upstream rilevante al freeze

Lo stato verificato del repository DwarfStar indica:

- DeepSeek V4.1 Flash è supportato su **Metal**;
- il modello V4.1 richiede un proprio GGUF/tokenizer/inference graph;
- il target `ds41f-q2` è già prodotto dal progetto;
- `ds41f-q2` occupa circa **341 GiB totali**;
- circa **152 GiB** sono "main weights";
- circa **189 GiB** sono Engram;
- Engram viene letto direttamente dal file secondo necessità e non viene reso residente;
- su una macchina Metal da 128 GB il Q2 funziona tramite SSD streaming;
- i backend non-Metal per V4.1 non sono ancora dichiarati implementati upstream.

Fonte:
https://github.com/antirez/ds4/blob/main/docs/MODELS.md

La configurazione ufficiale documentata per Metal è:

```bash
./download_model.sh ds41f-q2

./ds4 \
  -m gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --ssd-streaming \
  --ctx 32768
```

Per M1 sullo Spark partiremo con context molto più piccolo.

---

# 3. Scelta modello M1

## Usare il GGUF Q2 ufficiale DwarfStar

Non costruire il quant da zero durante M1.

Download:

```bash
./download_model.sh ds41f-q2
```

File atteso:

```text
gguf/DeepSeek-V4.1-Flash-Q2.gguf
```

Registrare:

```bash
ls -lh gguf/DeepSeek-V4.1-Flash-Q2.gguf
sha256sum gguf/DeepSeek-V4.1-Flash-Q2.gguf
```

Salvare SHA256 nel manifest M1.

### Motivo

Il Q2 V4.1 ufficiale è già:

- convertito con il layout che ds4 si aspetta;
- calibrato con imatrix;
- usato come reference Metal;
- comprensivo delle Engram table.

M1 non deve introdurre anche un nuovo problema di conversione/quantizzazione.

---

# 4. Cosa significa Q2 in questa milestone

Non assumere:

```text
tutto il modello = 2 bit
```

Il GGUF `ds41f-q2` usa una ricetta mixed-precision definita dal progetto.

Per M1 il termine **Q2** significa:

> usare **esattamente il GGUF `ds41f-q2` ufficiale**, senza modificarne le precisioni.

Non si deve:

- requantizzare;
- convertire gli expert FP4 a un formato nostro;
- creare una nuova ricetta Q2;
- alterare shared expert;
- alterare proiezioni;
- alterare Engram.

La milestone riguarda il **runtime**, non la qualità della quantizzazione.

---

# 5. M1 diverge deliberatamente dal MO generale

Il documento `MO_ds4_v41_GB10_implementation.md` descrive il progetto completo, compresa la persistent expert LRU (#647).

Per M1 cambiamo deliberatamente l'ordine.

## In M1 NON integriamo ancora la persistent LRU completa

Non servono ancora:

```text
d84d9f0  persistent LRU
e59f289  pooled resident cache buffers
ab98473  resident cache byte accounting
```

La ragione è metodologica:

> M1 deve dimostrare il path SSD completamente cold prima di introdurre il riuso cross-token.

Possiamo invece integrare la sola osservabilità di #647, se compatibile con il freeze:

```text
062d994  CUDA SSD streaming diagnostic counters
```

e la PR #1031:

```text
9b8d8fd  selected expert cache for IQ2/Q2 prefill
```

#1031 non è una persistent cache: corregge il selected-expert staging usato dal compute Q2.

---

# 6. Branch M1

Partire dal freeze definito nel MO.

Esempio:

```bash
cd ~/src/ds4-mo

git switch mo/gb10-v41
```

Se il branch MO contiene già la persistent #647 completa, per M1 è preferibile creare un branch direttamente dal tag di freeze:

```bash
git switch -c m1/v41-q2-cold mo-gb10-base-2026-09-13
```

Se il nome del tag è diverso:

```bash
git tag --list 'mo-gb10-base*'
```

Poi:

```bash
git status
git log --oneline -5
```

Deve essere pulito.

---

# 7. Freeze M1

Creare:

```text
M1_FREEZE.md
```

con:

```text
upstream base SHA
M1 branch SHA
CUDA version
driver version
kernel
Ubuntu version
nvcc version
GB10 identification
RAM
SSD model
SSD filesystem
model path
model SHA256
model size
PR SHA #1031
optional stats SHA
```

Comandi:

```bash
git rev-parse HEAD
uname -a
cat /etc/os-release
nvcc --version
nvidia-smi
lsblk -o NAME,MODEL,SIZE,FSTYPE,MOUNTPOINT
df -hT
sha256sum gguf/DeepSeek-V4.1-Flash-Q2.gguf
```

Commit:

```bash
git add M1_FREEZE.md
git commit -m "m1: freeze GB10 V4.1 Q2 cold-streaming baseline"
```

### M1-000

```text
m1: freeze GB10 V4.1 Q2 cold-streaming baseline
```

---

# 8. Baseline CUDA prima di V4.1

Prima del port V4.1 verificare che CUDA/GB10 funzioni con un modello già supportato.

Build:

```bash
make clean
make -j"$(nproc)" cuda-spark
```

Test disponibili:

```bash
./ds4_test
```

e, se presente nel freeze:

```bash
make test-cuda-q8-scratch
```

Usare anche un DeepSeek V4 Q2 già supportato:

```bash
./download_model.sh ds4f-q2
```

Smoke:

```bash
./ds4 \
  -m gguf/DeepSeek-V4-Flash-Q2.gguf \
  --ssd-streaming \
  --ssd-streaming-cold \
  --ctx 4096 \
  --temp 0 \
  --tokens 16 \
  --nothink \
  -p "Reply with exactly: CUDA BASELINE OK"
```

Se il nome effettivo del GGUF differisce, usare il file scaricato dal target `ds4f-q2`.

### Gate

```text
build      PASS
CUDA init  PASS
SSD path   PASS
generation PASS
```

Non procedere a V4.1 se la baseline V4 non è sana.

---

# 9. Osservabilità minima M1

M1 deve essere debuggabile.

Integrare, se compatibile con il freeze, il commit diagnostico della PR #647:

```bash
git cherry-pick 062d994
```

Obiettivo:

```text
DS4_CUDA_STREAM_STATS=1
```

deve consentire di osservare almeno:

- fetch calls;
- selected expert fetches;
- bytes letti dal file;
- eventuali cache hits;
- bytes serviti da cache.

Per M1 ci aspettiamo:

```text
persistent cache hits ~= 0
bytes from file       > 0
```

Se il commit non applica pulito perché upstream ha già incorporato statistiche equivalenti:

```text
NON forzare il cherry-pick
```

Creare invece un commit minimo che esponga gli stessi contatori.

Commit M1:

```text
M1-010  m1: add CUDA cold SSD streaming diagnostics
```

---

# 10. Selected-expert Q2 prefill

Integrare #1031 al SHA congelato:

```bash
git cherry-pick 9b8d8fd
```

Se il commit è già upstream nel freeze:

```bash
git log --all --oneline --grep='selected expert cache'
```

e non duplicarlo.

Scopo:

quando il selected-expert staging è valido, il Q2/IQ2 MMQ deve utilizzare:

```text
gate_ptr
up_ptr
down_ptr
compact selected slots
```

anziché riaprire le full routed-expert table.

### M1-020

```text
cuda: use selected expert cache for IQ2 prefill
```

### Gate regressione

Prima di V4.1:

```text
V4 Q2 SSD streaming continua a funzionare
```

---

# 11. Cold mode: definizione implementativa

Durante M1 il termine **cold** significa:

1. nessun preload iniziale di expert;
2. nessuna hotlist;
3. nessun seed;
4. nessuna persistent LRU cross-token;
5. ogni expert necessario deve poter essere recuperato dal backing GGUF;
6. selected staging temporaneo è consentito;
7. Engram resta su SSD;
8. page cache del kernel non viene considerata una expert cache logica.

Il flag upstream utile è:

```bash
--ssd-streaming-cold
```

La QA di ds4 usa esplicitamente questo flag per misurare il percorso cold e richiede che non si verifichino deadlock, missing expert o slowdown impossibili.

Fonte:
https://github.com/antirez/ds4/blob/main/QA_BEFORE_RELEASES.md

---

# 12. Non confondere selected staging con persistent cache

M1 permette questo:

```text
layer 12
router -> [4, 17, 31, 92, 211, 320]
          │
          ▼
SSD reads
          │
          ▼
compact selected staging
          │
          ▼
MMQ
```

M1 non richiede:

```text
token t
expert 17
   ↓
resident for token t+1
```

Quella sarà M2/cache.

Il selected staging può vivere abbastanza da completare il compute corrente, ma non deve essere usato per dichiarare M1 "warm".

---

# 13. V4.1 CUDA model dispatch

## M1-100

Primo commit V4.1:

```text
cuda-v41: add model dispatch and Q2 tensor metadata plumbing
```

Deve implementare:

- identificazione architecture V4.1;
- parsing metadata V4.1;
- tensor-name mapping;
- shape checks;
- quant type checks;
- Engram metadata;
- numero layer;
- routed expert count;
- top-k;
- V4.1-specific flags.

Non implementare ancora compute se non serve.

### Comportamento desiderato

```text
./ds4 -m V4.1-Q2.gguf --cuda
```

deve:

1. riconoscere il modello;
2. stampare metadata sensati;
3. allocare le strutture generali;
4. fallire esplicitamente alla prima primitive CUDA V4.1 non ancora implementata.

Non accettare:

```text
segfault
silent fallback
wrong architecture
V4 graph usato accidentalmente
```

---

# 14. Golden reference Metal

M1 deve avere un reference.

Usare lo **stesso Q2 GGUF** su Metal.

Per almeno tre prompt salvare:

```text
prompt
rendered token IDs
first N generated token IDs
generated text
```

Con:

```text
--temp 0
--nothink
```

e context piccolo.

Prompt A:

```text
Return exactly the number 42.
```

Prompt B:

```text
What is 17 + 25? Answer with only the number.
```

Prompt C:

```text
Write exactly: DeepSeek V4.1 test
```

Salvare come:

```text
reference/
  prompt_a.txt
  prompt_a.tokens
  prompt_a.output
  ...
```

### Perché token IDs

Il confronto testuale può nascondere:

- spazi;
- tokenizer edge;
- newline;
- rendering.

I token generati sono un reference più preciso.

---

# 15. Port V4.1 CUDA: ordine M1

Portare solo ciò che serve al **text greedy path Q2**.

Ordine:

```text
1. model layout
2. positional / RoPE
3. CED dataflow
4. CSA2 / sparse indexer
5. KV
6. router
7. Q2 selected expert compute
8. Engram
9. output projection / logits
```

Vision resta fuori.

DSpark resta fuori.

MTP resta fuori.

Server/concurrency resta fuori.

---

# 16. Primitive CUDA: commit piccoli

Usare una sequenza simile:

```text
M1-101  cuda-v41: port Q2 layout and dequant helpers
M1-102  cuda-v41: port V4.1 positional and rope path
M1-103  cuda-v41: implement CSA2 indexer path
M1-104  cuda-v41: implement CED text dataflow
M1-105  cuda-v41: implement V4.1 KV path
```

Ogni commit deve:

```text
compile
+
avere almeno uno scratch/unit test
+
non rompere V4 CUDA
```

Non creare:

```text
"implement V4.1 CUDA"
```

come singolo mega-commit.

---

# 17. Router V4.1

## M1-110

Commit:

```text
cuda-v41: implement top-6 routed expert selection
```

Registrare per debug:

```text
token
layer
top-6 expert IDs
top-6 scores
```

Il router deve essere testabile indipendentemente dal caricamento expert.

### Gate

Sui golden prompt:

```text
router top-6 dovrebbe coincidere con il reference
```

almeno sui layer/token di test.

Se una piccola differenza numerica cambia l'ordine di expert quasi equivalenti:

- registrarla;
- confrontare gli output successivi;
- non "aggiustare" il router con hack.

---

# 18. Q2 expert path M1

Il Q2 GGUF ufficiale deve usare i kernel/formati già supportati dal runtime DwarfStar, non una nuova quantizzazione.

Per il routed path M1 dobbiamo supportare il layout Q2/IQ2 del GGUF V4.1.

Il pattern deve essere:

```text
router
   ↓
top-6 IDs
   ↓
resolve ranges nel GGUF
   ↓
read only selected gate/up/down
   ↓
compact staging
   ↓
remap global ID -> compact slot
   ↓
Q2/IQ2 MMQ
   ↓
weighted reduction
```

### Non consentito

```text
map/load entire 384-expert table
```

per eseguire il layer.

---

# 19. M1-111 — selected expert loader V4.1

Commit:

```text
cuda-v41: map V4.1 Q2 selected experts from SSD
```

Implementare:

- tensor offset/range per `(layer, expert)`;
- gate;
- up;
- down;
- bounds checking;
- quant type validation;
- compact slot metadata.

### Debug log opzionale

```text
L=17 selected=[...]
expert=42 gate_off=...
expert=42 up_off=...
expert=42 down_off=...
bytes=...
```

Non tenerlo abilitato di default.

---

# 20. M1-112 — Q2 compute

Commit:

```text
cuda-v41: execute selected Q2 routed experts on GB10
```

Riutilizzare i kernel esistenti:

- IQ2_XXS;
- Q2_K;
- eventuali quant type effettivamente presenti nel Q2 ufficiale.

Non scrivere un nuovo kernel finché quelli esistenti possono rappresentare il tensor layout.

### Test sintetico

Prima del full model:

```text
known hidden vector
+
known selected experts
+
same Q2 weights
```

confrontare:

```text
CUDA output
vs
reference output
```

---

# 21. Engram M1

Engram è parte obbligatoria di M1.

Il Q2 file include circa:

```text
189 GiB Engram
```

e upstream Metal le legge direttamente dal file quando servono.

Non tentare di renderle residenti.

### Target

```text
token history
   ↓
ngram/hash
   ↓
row ID
   ↓
GGUF row location
   ↓
SSD read
   ↓
decode row
   ↓
V4.1 Engram projection/add
```

---

# 22. M1-120 — Engram hash/index parity

Commit:

```text
cuda-v41: implement Engram token-history and row-index parity
```

Prima dell'I/O GPU, verificare:

```text
token history -> same Engram row IDs as reference
```

Per una sequenza nota.

### Gate

```text
row IDs exact
```

Questo è un test deterministico; non accettare tolleranze.

---

# 23. M1-121 — Engram SSD reader

Commit:

```text
cuda-v41: add direct GGUF Engram row reads for GB10
```

M1 privilegia correttezza.

Non serve ancora:

- sophisticated page cache;
- async predictor;
- row popularity cache;
- batching aggressivo;
- prefetch.

Basta:

```text
resolve row
read row
validate bounds
return row
```

### Gate

- nessuna lettura fuori range;
- row bytes corretti;
- nessun tentativo di mappare 189 GiB in memoria residente come working set fisso.

---

# 24. M1-122 — Engram CUDA integrate

Commit:

```text
cuda-v41: integrate Engram contribution into text graph
```

Confrontare:

```text
Engram output Metal/reference
vs
CUDA
```

su alcuni layer/token.

Salvare:

```text
max_abs_error
mean_abs_error
```

---

# 25. KV e context M1

Partire con:

```text
--ctx 512
```

poi:

```text
--ctx 1024
```

poi:

```text
--ctx 4096
```

Non partire da 32K.

La documentazione upstream indica che la V4.1 KV globale è molto compatta, ma per M1 vogliamo minimizzare:

- allocazioni;
- debugging surface;
- session state;
- prefill time.

Il primo token corretto a `ctx=512` vale molto più di un crash a 32K.

---

# 26. Prima esecuzione end-to-end

Quando model graph, router, Q2 experts ed Engram sono collegati:

```bash
DS4_CUDA_STREAM_STATS=1 \
./ds4 \
  -m gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --ssd-streaming \
  --ssd-streaming-cold \
  --ctx 512 \
  --temp 0 \
  --tokens 1 \
  --nothink \
  --power 100 \
  -p "Return exactly the number 42."
```

### Primo obiettivo

Non 0,5 tok/s.

Il primo obiettivo è:

```text
1 token generato
```

senza:

- crash;
- illegal access;
- OOM;
- missing tensor;
- missing expert;
- unsupported quant;
- invalid Engram row;
- deadlock.

Tag provvisorio:

```bash
git tag -a m1-v41-q2-first-token \
  -m "First DeepSeek V4.1 Q2 token on GB10 cold SSD streaming"
```

---

# 27. Run 8 token

Poi:

```bash
DS4_CUDA_STREAM_STATS=1 \
./ds4 \
  -m gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --ssd-streaming \
  --ssd-streaming-cold \
  --ctx 512 \
  --temp 0 \
  --tokens 8 \
  --nothink \
  --power 100 \
  -p "Return exactly the number 42."
```

Gate:

```text
8 token senza errore
```

Se l'answer termina prima, usare un prompt che produca almeno 8 token.

---

# 28. Run 32 token

Poi:

```bash
DS4_CUDA_STREAM_STATS=1 \
./ds4 \
  -m gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --ssd-streaming \
  --ssd-streaming-cold \
  --ctx 1024 \
  --temp 0 \
  --tokens 32 \
  --nothink \
  --power 100 \
  -p "In four short sentences, describe how rain forms."
```

Questo è il primo test sufficiente a verificare:

- continuità decode;
- expert selection ripetuta;
- Engram state;
- KV progression;
- SSD reads ripetuti.

---

# 29. Cold-path instrumentation obbligatoria

Durante il run salvare:

```text
model load time
prefill time
prefill tok/s
first token latency
decode tokens
decode time
decode tok/s
selected expert fetch count
bytes read from expert backing file
Engram lookup count
Engram bytes read
cache hits
cache bytes
peak unified memory
SSD read bandwidth
```

Per M1:

```text
cache hits devono essere zero
```

oppure chiaramente attribuibili solo a strutture temporanee che non costituiscono persistent expert cache.

Se la build frozen contiene già una cache implicita, aggiungere un debug bypass M1.

---

# 30. Eventuale bypass cache M1

Se il CUDA upstream congelato ha ormai una persistent cache attiva anche senza #647, aggiungere un'opzione **solo diagnostica**:

```text
DS4_CUDA_FORCE_COLD_EXPERTS=1
```

Comportamento:

```text
skip persistent cache lookup
skip cache install
skip seed/preload
load selected expert from backing GGUF
```

Non deve disabilitare:

```text
compact selected staging
```

Commit:

```text
m1: add deterministic force-cold routed expert mode
```

Questa opzione serve solo per:

- test;
- baseline;
- profiling.

Non è una feature utente finale.

---

# 31. Cosa deve restare residente

M1 non significa "tutto da SSD".

La strategia è:

```text
resident:
- mandatory non-routed weights che entrano nel budget
- CUDA graph/runtime data
- router / small dense state
- KV
- scratch
- temporary selected-expert staging

SSD:
- routed experts che non sono residenti
- Engram
```

Per M1 non ottimizziamo il placement.

Usiamo il placement più semplice che:

```text
non supera 128 GB
```

e permette di generare.

---

# 32. Memory safety gate

Il DGX Spark ha 128 GB unified memory.

M1 fallisce se il sistema:

- entra sistematicamente in OOM;
- porta il sistema in swap thrash;
- lascia < headroom sufficiente al runtime;
- cresce indefinitamente ad ogni token;
- accumula expert invece di liberarli.

Registrare almeno:

```bash
free -h
nvidia-smi
```

prima e durante run.

Se possibile registrare anche:

```text
peak resident bytes
peak CUDA allocations
temporary selected-expert bytes
```

### M1 non richiede massima occupazione della RAM

Anzi, per il cold path è preferibile una configurazione conservativa.

---

# 33. Test di "nessuna crescita per token"

Eseguire:

```text
1 token
8 token
32 token
64 token
```

e registrare peak memory.

La memoria non deve crescere linearmente con il numero di token a causa degli expert letti.

Atteso:

```text
working set temporaneo oscillante/stabile
```

non:

```text
+expert bytes per ogni token
```

Questo test individua immediatamente un accidental persistent accumulation.

---

# 34. Correttezza greedy

Per M1 il test principale è:

```text
same GGUF
same prompt
same tokenizer
temperature = 0
thinking = off
```

Reference:

```text
Metal V4.1 Q2
```

Target:

```text
GB10 CUDA V4.1 Q2
```

Per i prompt corti:

```text
generated token IDs
```

devono idealmente coincidere.

Se non coincidono:

1. confrontare logits primo token;
2. confrontare router top-6;
3. confrontare Engram row IDs;
4. confrontare hidden output layer per layer.

Non continuare a ottimizzare prima di capire la divergenza.

---

# 35. Layer-by-layer debug ladder

Se il primo token differisce, usare questo ordine:

```text
embedding
↓
V4.1 position/RoPE
↓
encoder/CED block
↓
CSA2/indexer
↓
router logits
↓
top-6 expert IDs
↓
selected expert output
↓
Engram contribution
↓
decoder/CED block
↓
KV-derived attention
↓
final norm
↓
output projection
↓
logits
```

Per ogni punto salvare un hash o un piccolo dump.

Non dumpare tensor enormi completi se non necessario.

---

# 36. Throughput M1

Il performance target minimo è volutamente basso.

## Gate

Dopo che il modello è corretto, un run da almeno 16 token deve produrre:

```text
decode average >= 0,5 tok/s
```

ovvero:

```text
<= ~2 secondi per token in media
```

Il calcolo esclude:

- download;
- model startup;
- initial mapping.

Separare:

```text
prefill
```

da:

```text
decode
```

### Se fa 0,4 tok/s ma è corretto?

Non dichiarare M1 completa.

Prima verificare se il collo di bottiglia è un errore evidente:

- full table read;
- doppia lettura;
- accidental remap;
- cache/file mapping patologico;
- sync inutile.

Il target 0,5 tok/s non richiede ottimizzazione sofisticata, ma evita di accettare un path manifestamente rotto.

---

# 37. Nessun target di prefill aggressivo

M1 non impone:

```text
100 tok/s
500 tok/s
1000 tok/s
```

Il prefill deve soltanto:

- terminare;
- essere corretto;
- non mappare inutilmente l'intera MoE table;
- non esplodere in memoria.

#1031 è utile proprio perché evita una parte di questo comportamento nel Q2/IQ2 selected-prefill path.

---

# 38. Test SSD reale

Il modello deve essere sul vero NVMe locale dello Spark.

Non validare M1 con:

- network filesystem;
- NFS;
- SMB;
- FUSE remoto;
- storage cloud montato;
- tmpfs.

Registrare:

```bash
findmnt -T gguf/DeepSeek-V4.1-Flash-Q2.gguf
lsblk -o NAME,MODEL,SIZE,ROTA,FSTYPE,MOUNTPOINT
```

M1 è una milestone **SSD streaming**, quindi il backing store deve essere esplicito.

---

# 39. Test file-cache controlled

Per distinguere:

```text
SSD cold
```

da:

```text
Linux page-cache warm
```

fare almeno due categorie di run.

### A — process cold

Nuovo processo, `--ssd-streaming-cold`.

### B — repeated process

Nuovo processo ma file potenzialmente ancora in page cache.

Non tentare necessariamente di drop-caches in produzione.

Basta etichettare correttamente i risultati:

```text
OS page cache state = unknown/warm
expert persistent cache = disabled
```

Il concetto "all cold" di M1 riguarda la **expert cache logica ds4**, non il controllo assoluto della cache del kernel.

---

# 40. Test matrix M1

| Test | Context | Gen | Cache | Scopo |
|---|---:|---:|---|---|
| T0 | — | — | — | build |
| T1 | 512 | 1 | cold | first token |
| T2 | 512 | 8 | cold | short decode |
| T3 | 1024 | 32 | cold | repeated expert/Engram |
| T4 | 4096 | 32 | cold | context scaling |
| T5 | 1024 | 64 | cold | memory stability |
| T6 | 1024 | 16+ | cold | >=0.5 tok/s |
| T7 | 1024 | 16 | cold | greedy parity |
| T8 | 1024 | 16 | cold | restart reproducibility |

---

# 41. T0 — build gate

```bash
make clean
make -j"$(nproc)" cuda-spark
```

PASS solo se:

```text
zero build errors
```

Warnings nuovi introdotti dal port vanno classificati.

Non ignorare warning:

- signed/unsigned su offsets;
- pointer truncation;
- CUDA API return ignored;
- alignment;
- out-of-range enum quant.

---

# 42. T1 — first token

```bash
DS4_CUDA_STREAM_STATS=1 \
./ds4 \
  -m gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --ssd-streaming \
  --ssd-streaming-cold \
  --ctx 512 \
  --tokens 1 \
  --temp 0 \
  --nothink \
  --power 100 \
  -p "Return exactly the number 42."
```

PASS:

```text
process exit clean
token generated
no OOM
no illegal memory access
no unsupported tensor
no missing expert
Engram path completes
```

---

# 43. T2 — 8 token

Stesso setup:

```text
tokens = 8
```

PASS:

```text
decode advances 8 steps
KV advances correctly
no memory growth runaway
```

---

# 44. T3 — 32 token

Prompt:

```text
Explain briefly why water freezes.
```

PASS:

```text
32-token generation or EOS
no stall/deadlock
expert fetches continue
Engram continues
```

---

# 45. T4 — context 4096

Solo dopo T1-T3.

```bash
--ctx 4096
```

Prompt 500–1000 token oppure un prompt-file controllato.

PASS:

```text
prefill completes
generation starts
no memory pressure failure
```

Non è necessario generare 100 token.

---

# 46. T5 — memory stability

Generare almeno 64 token.

Campionare memoria ogni secondo.

Esempio script esterno:

```bash
while true; do
    date +%s
    free -b | grep Mem
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
    sleep 1
done
```

Adattare se `nvidia-smi` sullo Spark non rappresenta tutta la unified memory nel modo desiderato.

PASS:

```text
no monotonic routed-expert accumulation
```

---

# 47. T6 — 0,5 tok/s gate

Run:

```text
>=16 generated tokens
```

Calcolare:

```text
decode_tok_s = generated_tokens / decode_seconds
```

PASS:

```text
decode_tok_s >= 0.50
```

Non includere il prefill.

Salvare raw timing.

---

# 48. T7 — greedy reference

Confrontare tre prompt con Metal.

Salvare:

```text
prompt
input tokens
output tokens
output text
```

PASS desiderato:

```text
identical generated token sequence
```

Se uno diverge, non dichiarare M1 finché la divergenza non è spiegata.

---

# 49. T8 — reproducibilità

Tre restart completi:

```text
run A
run B
run C
```

stesso:

```text
GGUF
prompt
temp=0
ctx
token count
```

PASS:

```text
A == B == C
```

a livello di generated token IDs.

---

# 50. Telemetria minima per run

Salvare in JSONL:

```json
{
  "git_sha": "...",
  "model_sha256": "...",
  "backend": "cuda-gb10",
  "model": "ds41f-q2",
  "ssd_streaming": true,
  "logical_expert_cache": "cold",
  "ctx": 1024,
  "prompt_tokens": 0,
  "generated_tokens": 0,
  "prefill_seconds": 0.0,
  "prefill_tok_s": 0.0,
  "decode_seconds": 0.0,
  "decode_tok_s": 0.0,
  "expert_fetches": 0,
  "expert_file_bytes": 0,
  "persistent_cache_hits": 0,
  "engram_lookups": 0,
  "engram_file_bytes": 0,
  "peak_memory_bytes": 0,
  "output_sha256": "..."
}
```

Aggiungere campi se disponibili.

Non rimuovere raw logs.

---

# 51. Directory risultati

Nel repo:

```text
m1-results/
  env/
  build/
  reference-metal/
  first-token/
  cold-8/
  cold-32/
  ctx4096/
  memory/
  throughput/
  parity/
```

Non committare:

- GGUF;
- dump tensor enormi.

Committare:

- log piccoli;
- JSONL;
- checksum;
- script;
- summary.

---

# 52. Commit map M1

Sequenza raccomandata:

```text
M1-000  m1: freeze GB10 V4.1 Q2 cold-streaming baseline

M1-010  m1: add CUDA cold SSD streaming diagnostics
M1-020  cuda: use selected expert cache for IQ2 prefill

M1-100  cuda-v41: add Q2 model dispatch and metadata plumbing
M1-101  cuda-v41: port Q2 layout and dequant helpers
M1-102  cuda-v41: port V4.1 positional and rope path
M1-103  cuda-v41: implement CSA2 indexer path
M1-104  cuda-v41: implement CED text dataflow
M1-105  cuda-v41: implement V4.1 KV path

M1-110  cuda-v41: implement top-6 routed expert selection
M1-111  cuda-v41: map V4.1 Q2 selected experts from SSD
M1-112  cuda-v41: execute selected Q2 routed experts on GB10

M1-120  cuda-v41: implement Engram row-index parity
M1-121  cuda-v41: add SSD-backed Engram row reads
M1-122  cuda-v41: integrate Engram contribution into text graph

M1-130  cuda-v41: complete Q2 cold decode dataflow
M1-131  cuda-v41: complete Q2 selected-expert prefill path

M1-140  test: add V4.1 Q2 cold SSD streaming integration tests
M1-141  bench: record GB10 M1 cold-streaming baseline
```

Se una primitive è già presente upstream dopo il freeze, non creare un commit fittizio.

Registrare nel changelog M1:

```text
already upstream
```

---

# 53. Commit discipline

Ogni commit deve:

1. compilare;
2. mantenere la baseline V4 sana;
3. avere uno scopo unico;
4. includere il test pertinente;
5. non includere cleanup non correlato.

Non mescolare:

```text
Engram
+
router
+
cache
+
formatting
```

nello stesso commit.

---

# 54. Regole di rollback

Se un commit introduce:

- divergence greedy;
- OOM;
- invalid memory;
- aumento inspiegabile di file bytes;
- full expert table read;
- deadlock;

fare:

```bash
git revert <sha>
```

o lavorare su branch sperimentale.

Non aggiungere workaround sopra un errore non compreso.

---

# 55. Cosa NON fare in M1

Non implementare:

## Persistent LRU

Fuori M1.

## Expert frequency cache

Fuori M1.

## Hotlist

Fuori M1.

## Sensitivity analysis

Fuori M1.

## Q2/Q3/Q4 per singolo expert

Fuori M1.

## Neural predictor

Fuori M1.

## Transition predictor

Fuori M1.

## Speculative prefetch

Fuori M1.

## MTP / DSpark

Fuori M1.

## Vision

Fuori M1.

## Multi-Spark

Fuori M1.

## Server concurrency

Fuori M1.

## Long context > 4096 come requisito

Fuori M1.

---

# 56. Cosa si può fare solo per debugging

Sono consentiti:

- dump router top-6;
- dump Engram row IDs;
- tensor hash;
- layer timing;
- expert I/O timing;
- synchronous reads;
- extra CUDA synchronizations;
- assert aggressivi.

M1 non deve essere elegante.

Deve essere:

```text
corretto
riproducibile
debuggabile
```

---

# 57. Nessuna ottimizzazione prematura dell'I/O

Per M1, se un expert viene letto con una chiamata sincrona e costa tempo:

```text
va bene
```

Purché:

```text
legga solo quello necessario
```

e non:

```text
l'intera expert table
```

Il throughput target di 0,5 tok/s lascia molto spazio per una prima implementazione semplice.

L'async I/O viene dopo.

---

# 58. Nessun requisito di 70–80% cache hit

In M1 il valore desiderato è l'opposto:

```text
persistent cache hit = 0
```

Vogliamo misurare il costo puro:

\[
ColdCost
\]

che diventerà il denominatore per M2:

\[
Speedup_{cache}
=
\frac{throughput_{warm}}{throughput_{cold}}
\]

---

# 59. Perché questa baseline è importante per M2

Se M1 ottiene:

```text
0.5 tok/s
```

con persistent cache disabilitata, poi M2 può misurare in modo pulito:

```text
20 GB LRU
32 GB LRU
48 GB LRU
64 GB LRU
80 GB LRU
```

e osservare:

```text
cache hit
SSD bytes/token
tok/s
```

Senza M1 non sapremmo se un guadagno deriva dalla cache o da un altro cambiamento del graph.

---

# 60. M1 acceptance checklist

M1 è **PASS** solo se tutte queste caselle sono vere:

```text
[ ] build cuda-spark PASS
[ ] V4 CUDA baseline non regressa
[ ] ds41f-q2 GGUF ufficiale verificato con SHA256
[ ] V4.1 architecture riconosciuta su CUDA
[ ] V4.1 text CED graph funzionante
[ ] CSA2/indexer funzionante
[ ] V4.1 KV funzionante
[ ] router top-6 funzionante
[ ] Q2 selected expert lookup dal GGUF funzionante
[ ] Q2 expert compute CUDA funzionante
[ ] Engram row IDs corretti
[ ] Engram SSD row reads funzionanti
[ ] Engram contribution integrato
[ ] selected-expert staging usato
[ ] nessun full MoE table fetch necessario per layer
[ ] persistent expert cache non necessaria
[ ] --ssd-streaming-cold funzionante
[ ] first token PASS
[ ] 8-token run PASS
[ ] 32-token run PASS
[ ] 64-token stability PASS
[ ] ctx=4096 smoke PASS
[ ] greedy output confrontato con reference
[ ] 3 restart deterministici PASS
[ ] no OOM
[ ] no deadlock
[ ] no illegal CUDA access
[ ] no monotonic expert-memory accumulation
[ ] decode >= 0.50 tok/s su run >=16 token
```

---

# 61. M1 failure conditions

M1 è **FAIL** se anche una sola delle seguenti condizioni rimane irrisolta:

```text
model starts but cannot generate
expert ID resolves to wrong file range
Engram row is wrong
Engram is silently skipped
full expert table must be resident
persistent cache is required for correctness
memory grows every token
output greedy is unstable across restart
CUDA graph corrupts V4 baseline
decode stalls indefinitely on misses
throughput remains <0.5 tok/s per un evidente path inefficiente
```

---

# 62. Performance floor: come interpretare 0,5 tok/s

Il floor non è un target finale.

Serve soltanto a distinguere:

```text
cold path lento ma plausibile
```

da:

```text
cold path patologicamente inefficiente
```

A 0,5 tok/s:

```text
1 token ≈ 2 s
16 token ≈ 32 s di decode
32 token ≈ 64 s di decode
```

Questa prestazione è sufficiente per:

- debugging;
- tracer;
- correctness testing;
- raccolta iniziale di routing data;
- sviluppo M2.

---

# 63. Cosa succede subito dopo M1

M2 potrà partire dalla baseline:

```text
Q2 cold SSD streaming
~0.5+ tok/s
correct
```

e integrare:

```text
#647 persistent LRU
+
byte-accurate cache budget
+
pooled buffers
+
cache stats
```

Poi misurare:

```text
cold -> warm
```

senza cambiare il model graph.

Questa separazione è intenzionale.

---

# 64. Tag finale M1

Quando tutti i gate sono verdi:

```bash
git tag -a m1-v41-q2-cold-complete \
  -m "M1: DeepSeek V4.1 Flash Q2 cold SSD streaming works on single DGX Spark GB10"
```

Salvare:

```bash
git show m1-v41-q2-cold-complete
git bundle create m1-v41-q2-cold-complete.bundle \
  m1-v41-q2-cold-complete
```

Il bundle è consigliato per congelare il milestone indipendentemente dalle PR upstream.

---

# 65. Deliverable M1

Alla chiusura devono esistere:

```text
1. branch/tag riproducibile
2. environment manifest
3. model SHA256
4. build instructions
5. first-token log
6. 32/64-token cold run
7. Metal/reference comparison
8. expert SSD counters
9. Engram SSD counters
10. memory stability log
11. throughput baseline
12. output checksum
```

Summary finale:

```text
MODEL
DeepSeek V4.1 Flash Q2

HOST
NVIDIA DGX Spark / GB10

MODE
CUDA
single device
text-only
SSD streaming
logical expert cache cold
Engram on SSD

CORRECTNESS
PASS/FAIL

DECODE
X.XX tok/s

PREFILL
X.XX tok/s

EXPERT SSD
X GB/token

ENGRAM SSD
X MB/token

PEAK MEMORY
X GiB

REFERENCE
Metal Q2 parity: PASS/FAIL
```

---

# 66. Decisione finale M1

La filosofia della milestone è:

```text
prima facciamolo funzionare
con il percorso più stupido possibile
```

Non ci interessa ancora se ogni token fa centinaia di cache miss.

Non ci interessa ancora se il disco è il collo di bottiglia.

Non ci interessa ancora se il prefill è lento.

Ci interessa dimostrare che:

```text
V4.1 Q2
+
CUDA GB10
+
Engram da SSD
+
routed expert da SSD
=
modello completo e corretto
```

senza dipendere da una cache intelligente.

Quando questo è vero a circa **0,5 tok/s o più**, abbiamo finalmente una baseline affidabile sulla quale ha senso costruire tutto il resto.

---

# 67. Fonti upstream rilevanti

## DwarfStar repository

https://github.com/antirez/ds4

## V4.1 model guide

https://github.com/antirez/ds4/blob/main/docs/MODELS.md

Punti rilevanti:

- `ds41f-q2` ≈ 341 GiB;
- main weights ≈ 152 GiB;
- Engram ≈ 189 GiB;
- Engram letto dal file on demand;
- Q2 SSD streaming già funzionante sul backend Metal;
- V4.1 richiede proprio graph/GGUF dedicati;
- non-Metal V4.1 non ancora implementato upstream al freeze.

## SSD streaming guide

https://github.com/antirez/ds4/blob/main/docs/SSD_STREAMING.md

Punti rilevanti:

- SSD streaming legge gli expert routed mancanti dal GGUF;
- `--ssd-streaming-cold` è previsto per misure controllate;
- generation è più sensibile ai cache miss del prefill;
- la presenza del flag non implica automaticamente supporto di ogni modello/layout.

## Release QA

https://github.com/antirez/ds4/blob/main/QA_BEFORE_RELEASES.md

Punto rilevante:

- il cold streaming deve essere verificato esplicitamente per assenza di deadlock, missing expert e slowdown impossibili.

## PR #1031

https://github.com/antirez/ds4/pull/1031

Punto rilevante:

- il prefill IQ2/Q2 deve usare il compact selected-expert staging quando disponibile, invece di accedere alle full expert table.

## PR #647

https://github.com/antirez/ds4/pull/647

Per M1 usiamo al massimo l'osservabilità; la persistent LRU completa viene rinviata a M2.

---

# 68. Regola di handoff M1 → M2

Non aprire M2 finché non esiste un tag:

```text
m1-v41-q2-cold-complete
```

che riproduce:

```text
>= 0.5 tok/s decode
+
greedy correctness
+
Engram SSD
+
Q2 expert SSD
+
nessuna persistent expert cache richiesta
```

M2 deve essere una pura ottimizzazione della baseline M1, non un completamento mascherato del port V4.1.

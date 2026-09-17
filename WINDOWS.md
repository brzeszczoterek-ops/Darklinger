# Uruchamianie DARKLINGERA na Windowsie

## Aktualny stan obsługi

DARKLINGER nie jest jeszcze natywną aplikacją Windows. Zalecanym środowiskiem jest
**Windows 10 w wersji 2004 (build 19041) lub nowszy albo Windows 11, WSL2 i
Ubuntu 24.04**. Dzięki temu rdzeń agenta działa w takim samym środowisku
linuksowym jak wersja deweloperska, a użytkownik nadal uruchamia go z Windows
Terminala albo skrótem na pulpicie.

W obecnej wersji pod WSL2:

| Funkcja | Stan |
| --- | --- |
| Rozmowa tekstowa i pamięć | obsługiwane |
| Lokalne modele GGUF przez `llama.cpp` | obsługiwane |
| Akceleracja NVIDIA/CUDA | obsługiwana po konfiguracji CUDA dla WSL |
| Narzędzia MCP i przeglądarka | obsługiwane; pierwsza instalacja wymaga sieci |
| Sandbox Bubblewrap | należy zweryfikować testem na konkretnym WSL |
| Mowa, Whisper, Kokoro/Piper i F2 | eksperymentalne, nieobjęte tą instrukcją |
| Fizyczny kill switch `Q+P+0` z `/dev/input` | nieobsługiwany przez WSL |
| Awaryjne `darklinger-control panic-all` | obsługiwane z drugiego terminala WSL |

Jeżeli Bubblewrap nie przejdzie testów, DARKLINGER odmówi uruchamiania
wygenerowanego kodu. Jest to zachowanie fail-closed, a nie zgoda na wykonanie go
bez izolacji.

## 1. Instalacja WSL2 i Ubuntu

Uruchom **PowerShell jako administrator** i wykonaj:

```powershell
wsl --install -d Ubuntu-24.04
wsl --update
```

Uruchom ponownie komputer, otwórz Ubuntu z menu Start i utwórz linuksową nazwę
użytkownika oraz hasło. Hasło nie będzie wyświetlane podczas wpisywania — to
normalne.

Sprawdź w PowerShellu, czy dystrybucja używa WSL2:

```powershell
wsl --list --verbose
```

Przy Ubuntu wartość `VERSION` powinna wynosić `2`. Jeżeli dystrybucja ma inną
nazwę lub `Ubuntu-24.04` nie występuje na liście, sprawdź dostępne nazwy:

```powershell
wsl --list --online
```

## 2. Pakiety wymagane w Ubuntu

Wszystkie kolejne polecenia, poza wyraźnie oznaczonymi jako PowerShell, wykonuj
w terminalu Ubuntu:

```bash
sudo apt update
sudo apt install -y \
  git build-essential cmake ninja-build \
  python3 python3-venv python3-pip \
  nodejs npm bubblewrap libcurl4-openssl-dev
```

Sprawdź wersje:

```bash
python3 --version
node --version
npm --version
npx --version
bwrap --version
```

DARKLINGER wymaga Pythona **3.12 lub nowszego**. Ubuntu 24.04 dostarcza Pythona
3.12. Jeżeli polecenie pokazuje starszą wersję, nie kontynuuj instalacji w tej
dystrybucji — zainstaluj Ubuntu 24.04 w WSL.

Serwer przeglądarki MCP wymaga Node.js **18 lub nowszego**. Jeśli `node
--version` pokazuje starszą wersję, zainstaluj aktualne wydanie LTS zgodnie z
oficjalną stroną Node.js podlinkowaną na końcu instrukcji.

## 3. Instalacja DARKLINGERA

Kod i modele najlepiej przechowywać w linuksowym systemie plików WSL, a nie w
`/mnt/c`, ponieważ operacje na wielu plikach są tam szybsze.

```bash
cd ~
git clone https://github.com/brzeszczoterek-ops/Darklinger.git
cd Darklinger
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
cp .env.example .env
chmod 600 .env
```

Po każdym otwarciu nowego terminala aktywuj środowisko poleceniem:

```bash
cd ~/DARKLINGER
source .venv/bin/activate
```

## 4. Budowa `llama.cpp`

### Wariant CPU

Ten wariant działa bez karty NVIDIA, ale generowanie dużym modelem może być
wolne.

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build
cmake --build build --config Release -j "$(nproc)"
test -x build/bin/llama-server && echo "llama-server: OK"
```

### Wariant NVIDIA/CUDA

Najpierw zainstaluj aktualny sterownik NVIDIA **w Windowsie**, zaktualizuj WSL i
sprawdź w Ubuntu:

```bash
nvidia-smi
nvcc --version
```

Narzędzie CUDA instaluj zgodnie z instrukcją NVIDIA dla WSL. Nie instaluj w WSL
linuksowego sterownika karty graficznej — WSL korzysta ze sterownika Windows.
Gdy `nvidia-smi` i `nvcc` działają, zbuduj wariant CUDA:

```bash
cd ~/llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j "$(nproc)"
test -x build/bin/llama-server && echo "llama-server CUDA: OK"
```

DARKLINGER automatycznie szuka pliku
`~/llama.cpp/build/bin/llama-server`. Przy innej lokalizacji wpisz do `.env`
pełną ścieżkę linuksową, na przykład:

```dotenv
LLAMA_CPP_SERVER=/home/twoja_nazwa/llama.cpp/build/bin/llama-server
```

## 5. Dodanie modelu GGUF

Utwórz katalog modeli i otwórz go w Eksploratorze Windows:

```bash
mkdir -p ~/models
cd ~/models
explorer.exe .
```

Skopiuj do tego katalogu model z rozszerzeniem `.gguf`. Model może być czytany
bezpośrednio z dysku Windows, na przykład z
`/mnt/c/Users/NAZWA/Downloads`, ale lokalizacja `~/models` zwykle daje lepszą
wydajność. Skopiowanie dużego GGUF oczywiście zajmie dodatkowe miejsce na dysku.

## 6. Pierwszy start

```bash
cd ~/DARKLINGER
source .venv/bin/activate
darklinger-ui
```

Przy pierwszym uruchomieniu DARKLINGER zapyta o katalog modeli. Podaj:

```text
/home/twoja_nazwa/models
```

Najpierw w terminalu wybierz model i profil. Po jego uruchomieniu DARKLINGER otworzy
lokalny interfejs w domyślnej przeglądarce pod adresem
`http://127.0.0.1:8765/`. Interfejs nie jest wystawiany do sieci lokalnej ani do
Internetu. Na pierwszy test warto użyć ostrożnych
ustawień:

- `Context size`: `8192`;
- `GPU layers`: `auto` dla działającego CUDA albo `0` dla samego CPU;
- `Batch size`: `512`;
- `Micro-batch size`: `256`;
- `Parallel slots`: `1`;
- `Flash attention`: `auto`;
- `Reasoning mode`: `off`;
- `Anti-repetition`: `balanced`;
- `KV cache K type`: `q8_0`;
- `KV cache V type`: `q8_0`;
- `Additional llama.cpp arguments`: pozostaw puste.

Nie istnieje jeden dobry profil dla każdego GGUF. Zużycie RAM/VRAM zależy od
rozmiaru modelu, jego kwantyzacji i kontekstu. Jeżeli brakuje pamięci, zacznij od
zmniejszenia kontekstu i batcha. Dopiero potem można przetestować cache `q4_0`,
jeżeli wybrany backend go obsługuje. Zwiększaj parametry dopiero po stabilnym
teście.

Po starcie zadaj proste pytanie, a potem sprawdź narzędzie, na przykład prosząc
V o odczytanie nagłówka `README.md`. Sam tekst modelu mówiący, że wykonał
narzędzie, nie jest dowodem wykonania — sprawdź widoczny wynik i dziennik sesji.
Klawisz `F2` rozpoczyna i kończy nagrywanie push-to-talk, jeżeli lokalna warstwa
mowy została skonfigurowana. Przełącznik `V SPEAKS` steruje odczytywaniem
odpowiedzi. Przytrzymanie `HOLD TO KILL` zatrzymuje interfejs, V oraz model zarządzany
przez DARKLINGERA.

Jeżeli potrzebny jest wyłącznie stary interfejs terminalowy, uruchom zamiast tego
`v-core`.

## 7. Skrót uruchamiający z Windowsa

Repozytorium zawiera plik [`DARKLINGER-WSL.cmd`](DARKLINGER-WSL.cmd). Skopiuj go na
pulpit Windows. Zakłada on, że repozytorium znajduje się w `~/DARKLINGER` w
domyślnej dystrybucji WSL.

Po dwukrotnym kliknięciu plik otworzy sesję WSL, aktywuje `.venv` i uruchomi
`darklinger-ui`. Jeżeli repozytorium znajduje się gdzie indziej, zmień ścieżkę
`~/DARKLINGER` wewnątrz pliku.

## 8. Sprawdzenie instalacji

Przed poważniejszą pracą uruchom:

```bash
cd ~/DARKLINGER
source .venv/bin/activate
pytest -q test/test_model_loader.py test/test_agent_runtime.py
pytest -q test/test_sandbox.py
```

Pierwsze polecenie sprawdza loader i główną pętlę agenta. Drugie weryfikuje, czy
Bubblewrap rzeczywiście izoluje kod w danym WSL. Nie ignoruj awarii testów
sandboxa, jeżeli planujesz generowanie lub uruchamianie narzędzi.

## 9. Zatrzymanie i awaryjny panic

Zwykły proces zatrzymasz przez `Ctrl+C`. W razie zadania autonomicznego otwórz
drugi terminal Ubuntu i wykonaj:

```bash
cd ~/DARKLINGER
source .venv/bin/activate
darklinger-control panic-all
```

Po sprawdzeniu sytuacji ponowne uzbrojenie mechanizmu wymaga jawnego polecenia:

```bash
darklinger-control reset-panic
```

Fizyczny watcher `Q+P+0` używa linuksowego `/dev/input/event*` i nie jest
obecnie odpowiednim mechanizmem dla WSL. Do czasu stworzenia natywnego watchera
Windows używaj drugiego terminala z `panic-all`.

## 10. Funkcje, których na razie nie włączamy

- Ustaw `DARKLINGER_OWNER_MONITOR=0`. Automatyczne okno monitora zakłada obecnie
  terminal linuksowy z `gnome-terminal`.
- Warstwa głosowa korzysta z PipeWire, `pw-record`, `pw-play` i linuksowej
  obsługi F2. WSLg może udostępniać część audio, ale pełny tor mikrofon–STT–TTS
  nie jest jeszcze testowanym profilem Windows.
- Nie przekazuj DARKLINGEROWI ścieżek `C:\...`. W WSL ten sam dysk ma postać
  `/mnt/c/...`.

## 11. Typowe problemy

### `llama-server` nie został znaleziony

```bash
test -x ~/llama.cpp/build/bin/llama-server
```

Jeżeli plik istnieje w innym miejscu, ustaw jego pełną ścieżkę w `.env` jako
`LLAMA_CPP_SERVER`.

### `npx` nie został znaleziony

```bash
sudo apt update
sudo apt install -y nodejs npm
```

Pierwsze uruchomienie serwerów MCP może pobrać pakiety npm. Do tej jednorazowej
instalacji potrzebne jest połączenie z internetem.

### Przeglądarka MCP nie ma Firefoksa

```bash
npx --yes playwright install firefox
```

Następnie uruchom DARKLINGERA ponownie.

### Model nie mieści się w VRAM

Zatrzymaj DARKLINGERA przez `Ctrl+C`, uruchom go ponownie, edytuj profil i zmniejsz
`Context size`, `Batch size` oraz `Micro-batch size`. Kwantyzacja K/V zmniejsza
pamięć cache, ale nie zmniejsza samych wag zapisanych w GGUF.

### `local model loader requires an interactive terminal`

Uruchom `darklinger-ui` (lub terminalowe `v-core`) bezpośrednio w Windows
Terminalu/Ubuntu albo przez
`DARKLINGER-WSL.cmd`. Nie przekierowuj standardowego wejścia podczas wyboru modelu.

### Test Bubblewrap nie przechodzi

Nie wyłączaj zabezpieczenia, żeby wymusić wykonanie kodu. Najpierw zaktualizuj
WSL (`wsl --update` w PowerShellu), zrestartuj go (`wsl --shutdown`) i ponów test.
Do czasu poprawnego wyniku traktuj generowane narzędzia jako niedostępne.

## 12. Aktualizacja DARKLINGERA

```bash
cd ~/DARKLINGER
git pull --ff-only
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q test/test_model_loader.py test/test_agent_runtime.py
```

Jeśli masz lokalne zmiany w kodzie, `git pull --ff-only` może odmówić
aktualizacji. Nie usuwaj ich na ślepo — najpierw sprawdź `git status`.

## Oficjalne materiały

- [Instalacja WSL — Microsoft](https://learn.microsoft.com/en-us/windows/wsl/install)
- [Konfiguracja środowiska WSL — Microsoft](https://learn.microsoft.com/en-us/windows/wsl/setup/environment)
- [CUDA on WSL — NVIDIA](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [Budowa llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- [Playwright MCP](https://github.com/microsoft/playwright-mcp)
- [Node.js — pobieranie i wersje LTS](https://nodejs.org/en/download/)

# SLE Engine

Экспериментальный чистый C++20 proof-kernel для Synthetic Logic Evolution.

Что сейчас реализовано:
- битовые векторы и точные операции AND/OR/XOR/NOT
- универсальный 3-входовый тернарный примитив по 8-битной маске
- обратимый GF(2) byte-mixer, порождаемый row-XOR операциями
- residual XOR correction
- hard logic contracts: forbidden filter и required verification
- минимальный BooleanCascade и end-to-end Engine

Что уже добавлено поверх ядра:
- register-level x86-64 JIT backend для BooleanCascade с исполняемым буфером
- AVX-512-oriented JIT path с patch-point metadata под `vpternlogq` и patch-in-place масок, плюс scalar fallback
- JIT-first trainer: mutate -> patch binary -> evaluate -> rollback без полной пересборки каскада
- MDL-style fitness (`L(H) + L(D|H)`) вместо чисто линейного штрафа за число гейтов
- минимальный synthesis backend первого уровня
- `OutputFitness - ComplexityPenalty`
- локальный bit-mask mutator для поиска масок каскада
- topology mutation hooks и полноценный MCTS-based symbolic topology search
- multi-gate synthesis API
- residual synthesis layer
- бинарный, Delta-Sigma и LFSR encoder layer
- perception layer для стохастических bitstream-датасетов из сырых scalar/telemetry/audio-like потоков
- HLC-aware fitness penalty на orchestration уровне
- unified full-engine orchestration API
- high-level framework API: Dataset / Trainer / FrameworkModel
- примеры синтеза XOR, residual-correction, full pipeline и framework demo

Что сознательно ещё не реализовано или реализовано в базовом виде:
- AVX-512 path уже выделен архитектурно и умеет patch-in-place immediate для gate mask, но на этой машине не валидирован рантаймом из-за отсутствия AVX-512
- topology/GP mutation пока не переведены в чистый binary patch loop без общих rebuild-решений для всех структурных мутаций, хотя поиск теперь идет через планирующий MCTS вместо чистого случайного hill-climb
- island decomposition
- bit momentum, gossip, diffusion restart
- богатая topology mutation стратегия, а не только базовые hook'и
- сложный многоуровневый intermediate pressure
- полноценный HLC-aware search penalty, а не только orchestration-level contract application

Причина: ядро уже цельное, но часть backend-механизмов всё ещё остаётся исследовательской и эвристической зоной.

Новые ориентиры:
- поиск по умолчанию переведен на `MonteCarloTreeSearch`, который умеет планировать цепочки мутаций через несколько плохих локальных шагов
- для сенсорного/сырого входа добавлен `perception.hpp`: теперь можно строить `TaskKind::StochasticBitstream` датасеты из числовых рядов, телеметрии и вероятностных потоков без внешнего нейросетевого энкодера

Сборка:
```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

# Обучение BAEM v3 на PokerBench — «максимум» (настоящий трансформер)

Конвейер обучает покерного агента на основе твоего движка BAEM v3 данными
PokerBench, используя **настоящий трансформер из `unified_ml`** (не MLP-fallback).
Всё запускается из командной консоли.

## Что сделано для «максимума»
1. **Настоящий трансформер.** `unified_ml` вендорён в `third_party/unified_ml`.
   Сборка автоматически включает `-DHAVE_UNIFIED_ML`, и `PokerHistoryTransformer`
   использует `transformer::TransformerEncoder` (embed_dim=64, multi-head attention,
   2 слоя по умолчанию) с батчевым обучением (~3–4k примеров/с).
2. **Все 64 признака на входе**, включая карманные карты (слоты `[52..63]`,
   патч энкодера) и борд/историю/позицию — `embed_dim` поднят с 32 до 64.
3. **Данные из структурированных CSV** (`*_game_scenario_information.csv`), а не из
   текстовых промптов: точные `holding/board/pot/correct_decision/история`.
   Покрытие близко к 100% и для preflop, и для postflop.
4. **Веса трансформера сериализуются** внутри `PokerHistoryTransformer.save_weights`
   (тег `UML1` + конфиг + параметры). Проверено: загрузка в свежий экземпляр даёт
   побитово идентичный инференс → веса напрямую заходят в `BAEMAgent`.

## Запуск (одна команда)
```bash
bash tools/run_all.sh                 # setup → скачать CSV → подготовить → собрать → обучить → оценить
```
Переменные:
```bash
STAGE=preflop bash tools/run_all.sh           # только префлоп (60k, быстро, ~8 мин)
STAGE=postflop EPOCHS=20 bash tools/run_all.sh# постфлоп (500k, ~часы)
LIMIT=100000 bash tools/run_all.sh            # ограничить обучающие строки
EPOCHS=40 LR=0.0005 BATCH=128 bash tools/run_all.sh
```
Результат — `agent_weights.bin`.

## Пошагово
1. `bash tools/setup.sh` — g++ (≥10) + pip (`datasets huggingface_hub pandas pyarrow`).
2. Скачать CSV: `python3 tools/download_pokerbench.py --repo RZ412/PokerBench`
   (печатает путь к снапшоту).
3. Подготовить (тип preflop/postflop определяется автоматически):
   ```bash
   python3 tools/prepare_pokerbench_csv.py --csv <preflop_train>.csv  --out data/prepared_train.txt
   python3 tools/prepare_pokerbench_csv.py --csv <postflop_train>.csv --out data/prepared_train.txt --append
   ```
   Перед полным прогоном полезно: `--inspect 3` (печатает разбор).
4. Сборка: `bash tools/build.sh` (соберёт `libuml.a` один раз, затем оба инструмента).
5. Обучение:
   ```bash
   ./bin/train_pokerbench --data data/prepared_train.txt --out agent_weights.bin \
       --epochs 30 --lr 5e-4 --batch 64
   ```
   train/val split 90/10, взвешивание классов, ранняя остановка, recall по классам.
6. Оценка/игра:
   ```bash
   ./bin/play_pokerbench --weights agent_weights.bin --eval data/prepared_test.txt
   echo "0 0 150 0 1 12 25 0 0" | ./bin/play_pokerbench --weights agent_weights.bin --stdin  # AA префлоп
   ```

## Интеграция в твой движок
```cpp
baem::BAEMConfig cfg;                 // transformer_cfg по умолчанию: embed_dim=64, ff=128, heads=4, layers=2
baem::BAEMAgent agent(std::move(oracle), nullptr, cfg);
agent.trainer().transformer().load_weights("agent_weights.bin");   // веса совместимы 1:1
// далее agent.decide(...) использует обученную policy
```
**Важно:** конфиг трансформера в обучении и в агенте должен совпадать
(embed_dim/ff_hidden/num_heads/num_layers). По умолчанию совпадает; load_weights
проверяет это по заголовку и вернёт false при несоответствии.

## Формат подготовленных данных (справка)
```
label street pot curbet hero_ip c1 c2 nboard [board...] nhist [type amt player]...
```
label/type 0..4 = Fold,Check,Call,Raise,AllIn; street 0..3; суммы в BB×100;
карты 0..51 (2c=0..Ac=12, 2d=13.., 2h=26.., 2s=39..).

## Ограничения и следующий шаг
Выход — 5 классов **типов** действий; размер ставки схлопывается в `Raise`
(этого требует совместимость с `poker::ActionType` и policy-массивом `[5]`
в `BAEMAgent`). CSV содержит точные размеры ставок, поэтому следующий разумный
апгрейд — добавить отдельную **голову размера ставки** (бакеты по доле пота),
не ломая 5-классовую политику типов. Скажи, если нужно — добавлю.

## Сборка без unified_ml
Если удалить `third_party/unified_ml`, `build.sh` соберёт MLP-fallback
(быстрее, но слабее). Интерфейсы и скрипты те же.

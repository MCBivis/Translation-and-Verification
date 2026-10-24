# HW4. Промежуточное представление и базовые оптимизации

Запуск тестов:

```bash
python funny_ir_opt.py
```

Запуск проверки файла .funny:

```bash
python funny_ir_opt.py program.funny
```

После запуска создаётся:

- `hw4_output.json` и `hw4_output.txt` — результаты.

Для получения графов:

```bash
python funny_ir_opt.py program.funny --dot-dir graphs
```

Для включения CSE:

```bash
python funny_ir_opt.py program.funny --cse
```

Оптимизации можно отключать независимо:
```bash
python funny_ir_opt.py program.funny --no-const-prop
```
```bash
python funny_ir_opt.py program.funny --no-algebra
```
```bash
python funny_ir_opt.py program.funny --no-dce
```
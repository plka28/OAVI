from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image

ORIGIN = "https://www.slavcorpora.ru"
SAMPLE_ID = "b008ae91-32cf-4d7d-84e4-996144e4edb7"

VARIANT = 12
OPERATOR_NAME = "Оператор Круна 3x3"
GRADIENT_FORMULA_NAME = "G = |Gx| + |Gy|"
IMAGE_INDEX = 4

# Порог подбирается опытным путем.
THRESHOLD = 110

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
SRC_DIR = BASE_DIR / "src"
REPORT_PATH = BASE_DIR / "report.md"


def fetch_image_paths(origin: str, sample_id: str) -> list[str]:
    response = requests.get(f"{origin}/api/samples/{sample_id}", timeout=30)
    response.raise_for_status()
    sample_data = response.json()
    return [f"{origin}/images/{page['filename']}" for page in sample_data["pages"]]


def download_image_rgb(image_url: str) -> np.ndarray:
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    return np.asarray(pil_image, dtype=np.uint8)


def save_rgb(image: np.ndarray, path: Path) -> None:
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB").save(path)


def save_gray(image: np.ndarray, path: Path) -> None:
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="L").save(path)


def cleanup_generated_files(directory: Path) -> None:
    for file_path in directory.glob("*"):
        if file_path.is_file():
            file_path.unlink()


def rgb_to_grayscale_weighted(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float64)
    gray = 0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2]
    return np.clip(gray, 0, 255).round().astype(np.uint8)


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("Размер ядра должен быть нечетным.")

    pad_h = kh // 2
    pad_w = kw // 2
    image_f = image.astype(np.float64)
    padded = np.pad(image_f, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, kernel, optimize=True)


def normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    data_min = float(data.min())
    data_max = float(data.max())
    if data_max <= data_min:
        return np.zeros_like(data, dtype=np.uint8)
    normalized = (data - data_min) * 255.0 / (data_max - data_min)
    return np.clip(normalized, 0, 255).round().astype(np.uint8)


def write_report(
    source_url: str,
    width: int,
    height: int,
) -> None:
    report = f"""# Лабораторная работа №4
## Выделение контуров на изображении

### Вариант {VARIANT}
- Оператор: {OPERATOR_NAME}
- Формула градиента: `{GRADIENT_FORMULA_NAME}`
- Порог бинаризации градиентной матрицы: `T={THRESHOLD}` (подобран опытным путем)

### Исходные данные
- Источник: `{source_url}`
- Размер изображения: `{width}x{height}`

### Формулы

Перевод цветного изображения в полутоновое:

```text
I(x, y) = 0.299 * R(x, y) + 0.587 * G(x, y) + 0.114 * B(x, y)
```

Градиенты по оператору Круна (ядра 3x3):

```text
Kx = [[ 17,   0, -17],
      [ 61,   0, -61],
      [ 17,   0, -17]]

Ky = [[ 17,  61,  17],
      [  0,   0,   0],
      [-17, -61, -17]]
```

```text
Gx = I * Kx
Gy = I * Ky
G  = |Gx| + |Gy|
```

Бинаризация градиентной матрицы:

```text
B(x, y) = 255, если G(x, y) >= T, иначе 0
```

### Результаты

#### 1. Исходное цветное изображение
![source](src/source_color.png)

#### 2. Полутоновое изображение
![gray](src/grayscale.bmp)

#### 3. Градиентные матрицы (нормализованные в диапазон 0..255)
| Gx | Gy | G |
|:--:|:--:|:--:|
| ![gx](src/gx_norm.bmp) | ![gy](src/gy_norm.bmp) | ![g](src/g_norm.bmp) |

#### 4. Бинаризованная градиентная матрица G
![binary](src/g_binary.bmp)

### Таблица файлов
| Операция | Файл |
|:---------|:-----|
| Исходное цветное | `src/source_color.png` |
| Полутоновое | `src/grayscale.bmp` |
| Нормализованная матрица Gx | `src/gx_norm.bmp` |
| Нормализованная матрица Gy | `src/gy_norm.bmp` |
| Нормализованная матрица G | `src/g_norm.bmp` |
| Бинаризация G | `src/g_binary.bmp` |

### Вывод
Для варианта {VARIANT} реализовано выделение контуров оператором Круна 3x3 с формулой `{GRADIENT_FORMULA_NAME}`. Получены требуемые промежуточные матрицы `Gx`, `Gy`, `G` и итоговая бинаризованная карта контуров.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_generated_files(RESULTS_DIR)
    cleanup_generated_files(SRC_DIR)

    image_paths = fetch_image_paths(ORIGIN, SAMPLE_ID)
    if not image_paths:
        raise RuntimeError("Список изображений пуст.")
    if IMAGE_INDEX < 0 or IMAGE_INDEX >= len(image_paths):
        raise IndexError("IMAGE_INDEX выходит за пределы списка image_paths.")

    source_url = image_paths[IMAGE_INDEX]
    source_rgb = download_image_rgb(source_url)
    gray = rgb_to_grayscale_weighted(source_rgb)

    kx = np.array(
        [
            [17, 0, -17],
            [61, 0, -61],
            [17, 0, -17],
        ],
        dtype=np.float64,
    )
    ky = np.array(
        [
            [17, 61, 17],
            [0, 0, 0],
            [-17, -61, -17],
        ],
        dtype=np.float64,
    )

    gx_raw = convolve2d(gray, kx)
    gy_raw = convolve2d(gray, ky)
    g_raw = np.abs(gx_raw) + np.abs(gy_raw)

    gx_norm = normalize_to_uint8(np.abs(gx_raw))
    gy_norm = normalize_to_uint8(np.abs(gy_raw))
    g_norm = normalize_to_uint8(g_raw)
    g_binary = np.where(g_norm >= THRESHOLD, 255, 0).astype(np.uint8)

    for out_dir in (RESULTS_DIR, SRC_DIR):
        save_rgb(source_rgb, out_dir / "source_color.png")
        save_gray(gray, out_dir / "grayscale.bmp")
        save_gray(gx_norm, out_dir / "gx_norm.bmp")
        save_gray(gy_norm, out_dir / "gy_norm.bmp")
        save_gray(g_norm, out_dir / "g_norm.bmp")
        save_gray(g_binary, out_dir / "g_binary.bmp")

    h, w = gray.shape
    write_report(source_url=source_url, width=w, height=h)

    print("Лабораторная работа №4 выполнена.")
    print(f"Вариант: {VARIANT}")
    print(f"Оператор: {OPERATOR_NAME}")
    print(f"Формула градиента: {GRADIENT_FORMULA_NAME}")
    print(f"Порог бинаризации: {THRESHOLD}")
    print(f"Результаты: {RESULTS_DIR}")
    print(f"Файлы для отчета: {SRC_DIR}")
    print(f"Отчет: {REPORT_PATH}")


if __name__ == "__main__":
    main()

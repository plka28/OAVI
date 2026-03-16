from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image

ORIGIN = "https://www.slavcorpora.ru"
SAMPLE_ID = "b008ae91-32cf-4d7d-84e4-996144e4edb7"
IMAGE_INDEX = 0

M = 3
N = 2
K = M / N

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


def rgb_to_hsi(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb_norm = rgb.astype(np.float64) / 255.0
    r = rgb_norm[..., 0]
    g = rgb_norm[..., 1]
    b = rgb_norm[..., 2]

    intensity = (r + g + b) / 3.0

    min_rgb = np.minimum(np.minimum(r, g), b)
    saturation = np.zeros_like(intensity)
    nonzero_intensity = intensity > 1e-12
    saturation[nonzero_intensity] = (
        1.0 - min_rgb[nonzero_intensity] / intensity[nonzero_intensity]
    )
    saturation = np.clip(saturation, 0.0, 1.0)

    numerator = 0.5 * ((r - g) + (r - b))
    denominator = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    denominator = np.where(denominator < 1e-12, 1e-12, denominator)
    theta = np.arccos(np.clip(numerator / denominator, -1.0, 1.0))
    hue = np.where(b <= g, theta, 2.0 * np.pi - theta)
    hue = np.mod(hue, 2.0 * np.pi)

    return hue, saturation, intensity


def hsi_to_rgb(hue: np.ndarray, saturation: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    h = np.mod(hue, 2.0 * np.pi)
    s = np.clip(saturation, 0.0, 1.0)
    i = np.clip(intensity, 0.0, 1.0)

    r = np.zeros_like(i)
    g = np.zeros_like(i)
    b = np.zeros_like(i)

    eps = 1e-12
    sector1 = (h >= 0) & (h < 2.0 * np.pi / 3.0)
    sector2 = (h >= 2.0 * np.pi / 3.0) & (h < 4.0 * np.pi / 3.0)
    sector3 = ~(sector1 | sector2)

    h1 = h[sector1]
    i1 = i[sector1]
    s1 = s[sector1]
    b[sector1] = i1 * (1.0 - s1)
    den1 = np.cos(np.pi / 3.0 - h1)
    den1 = np.where(np.abs(den1) < eps, eps, den1)
    r[sector1] = i1 * (1.0 + s1 * np.cos(h1) / den1)
    g[sector1] = 3.0 * i1 - (r[sector1] + b[sector1])

    h2 = h[sector2] - 2.0 * np.pi / 3.0
    i2 = i[sector2]
    s2 = s[sector2]
    r[sector2] = i2 * (1.0 - s2)
    den2 = np.cos(np.pi / 3.0 - h2)
    den2 = np.where(np.abs(den2) < eps, eps, den2)
    g[sector2] = i2 * (1.0 + s2 * np.cos(h2) / den2)
    b[sector2] = 3.0 * i2 - (r[sector2] + g[sector2])

    h3 = h[sector3] - 4.0 * np.pi / 3.0
    i3 = i[sector3]
    s3 = s[sector3]
    g[sector3] = i3 * (1.0 - s3)
    den3 = np.cos(np.pi / 3.0 - h3)
    den3 = np.where(np.abs(den3) < eps, eps, den3)
    b[sector3] = i3 * (1.0 + s3 * np.cos(h3) / den3)
    r[sector3] = 3.0 * i3 - (g[sector3] + b[sector3])

    rgb = np.stack([r, g, b], axis=-1)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def bilinear_resize(image: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    src_h, src_w, channels = image.shape
    if new_h <= 0 or new_w <= 0:
        raise ValueError("Размер выходного изображения должен быть положительным.")

    if src_h == new_h and src_w == new_w:
        return image.copy()

    image_f = image.astype(np.float64)
    y = np.linspace(0, src_h - 1, new_h) if new_h > 1 else np.array([0.0])
    x = np.linspace(0, src_w - 1, new_w) if new_w > 1 else np.array([0.0])

    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)

    wy = y - y0
    wx = x - x0

    wa = (1.0 - wy)[:, None] * (1.0 - wx)[None, :]
    wb = wy[:, None] * (1.0 - wx)[None, :]
    wc = (1.0 - wy)[:, None] * wx[None, :]
    wd = wy[:, None] * wx[None, :]

    top_left = image_f[y0[:, None], x0[None, :]]
    bottom_left = image_f[y1[:, None], x0[None, :]]
    top_right = image_f[y0[:, None], x1[None, :]]
    bottom_right = image_f[y1[:, None], x1[None, :]]

    result = (
        top_left * wa[..., None]
        + bottom_left * wb[..., None]
        + top_right * wc[..., None]
        + bottom_right * wd[..., None]
    )
    return np.clip(result, 0, 255).round().astype(np.uint8)


def stretch_interpolation(image: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 0:
        raise ValueError("Коэффициент растяжения должен быть положительным.")
    src_h, src_w, _ = image.shape
    new_h = max(1, int(round(src_h * factor)))
    new_w = max(1, int(round(src_w * factor)))
    return bilinear_resize(image, new_h, new_w)


def decimation(image: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 0:
        raise ValueError("Коэффициент сжатия должен быть положительным.")
    return image[::factor, ::factor].copy()


def one_pass_resample(image: np.ndarray, factor: float) -> np.ndarray:
    if factor <= 0:
        raise ValueError("Коэффициент передискретизации должен быть положительным.")
    src_h, src_w, _ = image.shape
    new_h = max(1, int(round(src_h * factor)))
    new_w = max(1, int(round(src_w * factor)))
    return bilinear_resize(image, new_h, new_w)


def write_report(
    image_url: str,
    image_shape: tuple[int, int, int],
    stretched_shape: tuple[int, int, int],
    decimated_shape: tuple[int, int, int],
    two_pass_shape: tuple[int, int, int],
    one_pass_shape: tuple[int, int, int],
) -> None:
    report = f"""# Лабораторная работа №1
## Цветовые модели и передискретизация изображений

### Исходное изображение

![Исходное изображение](src/source.png)

### 1. Цветовые модели

#### 1.1 Компоненты R, G, B

|             Красный канал              |             Зеленый канал              |              Синий канал               |
|:--------------------------------------:|:--------------------------------------:|:--------------------------------------:|
| ![R](src/r_channel.png) | ![G](src/g_channel.png) | ![B](src/b_channel.png) |

#### 1.2 Яркостная компонента HSI

![Яркостная компонента HSI](src/intensity_channel.png)

#### 1.3 Инвертирование яркостной компоненты

|            Исходное изображение            |                  С инвертированной яркостью                   |
|:------------------------------------------:|:-------------------------------------------------------------:|
| ![Исходное](src/source.png) | ![Инвертированное](src/inverted_intensity.png) |

### 2. Передискретизация (M={M}, N={N}, K={K:.3f})

#### 2.1 Растяжение в M раз (метод билинейной интерполяции)

|                  Исходное                  |                   Растянутое                   |
|:------------------------------------------:|:----------------------------------------------:|
| ![Исходное](src/source.png) | ![Растянутое](src/upscaled.png) |

#### 2.2 Сжатие в N раз (метод прореживания)

|                  Исходное                  |                    Сжатое                    |
|:------------------------------------------:|:--------------------------------------------:|
| ![Исходное](src/source.png) | ![Сжатое](src/downscaled.png) |

#### 2.3 Двухпроходная передискретизация (растяжение + сжатие)

|                  Исходное                  |             Результат двух проходов             |
|:------------------------------------------:|:-----------------------------------------------:|
| ![Исходное](src/source.png) | ![Два прохода](src/two_pass.png) |

#### 2.4 Однопроходная передискретизация (прямое масштабирование)

|                  Исходное                  |            Результат одного прохода             |
|:------------------------------------------:|:-----------------------------------------------:|
| ![Исходное](src/source.png) | ![Один проход](src/one_pass.png) |

### Результаты выполнения

| Операция                          | Размер изображения |
|:----------------------------------|-------------------:|
| Исходное изображение              | {image_shape[1]}x{image_shape[0]} |
| Растяжение (M={M})                | {stretched_shape[1]}x{stretched_shape[0]} |
| Сжатие (N={N})                    | {decimated_shape[1]}x{decimated_shape[0]} |
| Двухпроходная (M={M} + N={N})     | {two_pass_shape[1]}x{two_pass_shape[0]} |
| Однопроходная (K={K:.3f})         | {one_pass_shape[1]}x{one_pass_shape[0]} |

### Выводы

В ходе выполнения лабораторной работы были изучены:

1. **Цветовые модели RGB и HSI**:
   - Выделены компоненты красного, зеленого и синего каналов.
   - Выполнено преобразование RGB -> HSI.
   - Произведено инвертирование яркостной компоненты.

2. **Методы передискретизации**:
   - Реализован метод билинейной интерполяции для растяжения изображения.
   - Реализован метод прореживания для сжатия изображения.
   - Выполнена двухпроходная передискретизация: растяжение в M раз с последующим сжатием в N раз.
   - Реализована однопроходная передискретизация: прямое масштабирование в K=M/N раз.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SRC_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = fetch_image_paths(ORIGIN, SAMPLE_ID)
    if not image_paths:
        raise RuntimeError("Список изображений пуст.")
    if IMAGE_INDEX < 0 or IMAGE_INDEX >= len(image_paths):
        raise IndexError("IMAGE_INDEX выходит за пределы списка image_paths.")

    selected_image_url = image_paths[IMAGE_INDEX]
    source = download_image_rgb(selected_image_url)

    save_rgb(source, RESULTS_DIR / "source.png")
    save_rgb(source, SRC_DIR / "source.png")

    # 1) Цветовые модели.
    r_comp = np.zeros_like(source)
    g_comp = np.zeros_like(source)
    b_comp = np.zeros_like(source)
    r_comp[..., 0] = source[..., 0]
    g_comp[..., 1] = source[..., 1]
    b_comp[..., 2] = source[..., 2]
    save_rgb(r_comp, RESULTS_DIR / "component_r.png")
    save_rgb(g_comp, RESULTS_DIR / "component_g.png")
    save_rgb(b_comp, RESULTS_DIR / "component_b.png")
    save_rgb(r_comp, SRC_DIR / "r_channel.png")
    save_rgb(g_comp, SRC_DIR / "g_channel.png")
    save_rgb(b_comp, SRC_DIR / "b_channel.png")

    h, s, intensity = rgb_to_hsi(source)
    intensity_img = np.clip(intensity * 255.0, 0, 255).round().astype(np.uint8)
    save_gray(intensity_img, RESULTS_DIR / "intensity_hsi.png")
    save_gray(intensity_img, SRC_DIR / "intensity_channel.png")

    inverted_intensity = 1.0 - intensity
    inverted_rgb = hsi_to_rgb(h, s, inverted_intensity)
    save_rgb(inverted_rgb, RESULTS_DIR / "intensity_inverted_rgb.png")
    save_rgb(inverted_rgb, SRC_DIR / "inverted_intensity.png")

    # 2) Передискретизация.
    stretched = stretch_interpolation(source, M)
    save_rgb(stretched, RESULTS_DIR / "stretch_m.png")
    save_rgb(stretched, SRC_DIR / "upscaled.png")

    decimated = decimation(source, N)
    save_rgb(decimated, RESULTS_DIR / "decimation_n.png")
    save_rgb(decimated, SRC_DIR / "downscaled.png")

    two_pass = decimation(stretched, N)
    save_rgb(two_pass, RESULTS_DIR / "resample_two_pass.png")
    save_rgb(two_pass, SRC_DIR / "two_pass.png")

    one_pass = one_pass_resample(source, K)
    save_rgb(one_pass, RESULTS_DIR / "resample_one_pass.png")
    save_rgb(one_pass, SRC_DIR / "one_pass.png")

    write_report(
        image_url=selected_image_url,
        image_shape=source.shape,
        stretched_shape=stretched.shape,
        decimated_shape=decimated.shape,
        two_pass_shape=two_pass.shape,
        one_pass_shape=one_pass.shape,
    )

    print("Лабораторная работа выполнена.")
    print(f"Изображения сохранены в: {RESULTS_DIR}")
    print(f"Изображения по шаблону сохранены в: {SRC_DIR}")
    print(f"Отчет сохранен в: {REPORT_PATH}")
    print(f"Использованное изображение: {selected_image_url}")


if __name__ == "__main__":
    main()

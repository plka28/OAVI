from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image

ORIGIN = "https://www.slavcorpora.ru"
SAMPLE_ID = "b008ae91-32cf-4d7d-84e4-996144e4edb7"

VARIANT = 12
WINDOW_SIZES = [3, 25]
K_PARAM = -0.1

# Несколько изображений для демонстрации "до/после" на разных входных данных.
IMAGE_INDICES = [0, 5, 10]

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
SRC_DIR = BASE_DIR / "src"
REPORT_PATH = BASE_DIR / "report.md"


@dataclass
class CaseResult:
    case_no: int
    image_index: int
    source_url: str
    width: int
    height: int
    source_name: str
    gray_name: str
    binary_names: dict[int, str]


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


def rgb_to_grayscale_weighted(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float64)
    gray = 0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2]
    return np.clip(gray, 0, 255).round().astype(np.uint8)


def local_mean_sqmean(gray: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    if window_size % 2 == 0 or window_size < 3:
        raise ValueError("Размер окна должен быть нечетным и >= 3.")

    gray_f = gray.astype(np.float64)
    radius = window_size // 2

    padded = np.pad(gray_f, ((radius, radius), (radius, radius)), mode="edge")
    padded_sq = padded * padded

    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    integral_sq = np.pad(padded_sq, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)

    h, w = gray.shape
    y0 = np.arange(h)[:, None]
    x0 = np.arange(w)[None, :]
    y1 = y0 + window_size
    x1 = x0 + window_size

    local_sum = (
        integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    )
    local_sum_sq = (
        integral_sq[y1, x1]
        - integral_sq[y0, x1]
        - integral_sq[y1, x0]
        + integral_sq[y0, x0]
    )

    pixels_in_window = float(window_size * window_size)
    mean = local_sum / pixels_in_window
    sqmean = local_sum_sq / pixels_in_window
    return mean, sqmean


def nick_binarization(gray: np.ndarray, window_size: int, k_value: float) -> np.ndarray:
    mean, sqmean = local_mean_sqmean(gray, window_size)
    variance = np.maximum(sqmean - mean * mean, 0.0)

    # NICK:
    # T(x, y) = m(x, y) + k * sqrt(var(x, y) + sqmean(x, y))
    threshold = mean + k_value * np.sqrt(np.maximum(variance + sqmean, 0.0))
    return np.where(gray.astype(np.float64) > threshold, 255, 0).astype(np.uint8)


def cleanup_generated_files(directory: Path) -> None:
    for file_path in directory.glob("img*"):
        if file_path.is_file():
            file_path.unlink()


def write_report(cases: list[CaseResult]) -> None:
    window_label = ", ".join(f"{w}x{w}" for w in WINDOW_SIZES)
    lines: list[str] = []
    lines.append("# Лабораторная работа №2")
    lines.append("## Обесцвечивание и бинаризация растровых изображений")
    lines.append("")
    lines.append(f"### Вариант {VARIANT}: адаптивная бинаризация NICK")
    lines.append(f"### Окна по обновленному PDF: {window_label}")
    lines.append("")
    lines.append("### Исходные данные")
    lines.append(f"- Количество изображений: {len(cases)}")
    lines.append(f"- Параметр метода NICK: `k={K_PARAM}`")
    lines.append(f"- Размеры окон NICK: `{window_label}`")
    lines.append("- Формат исходных локальных изображений: PNG (`src/img*_source.png`)")
    lines.append("- Формат полутоновых и бинарных изображений: BMP")
    lines.append("")
    lines.append("### Формулы")
    lines.append("")
    lines.append("Обесцвечивание (взвешенное усреднение RGB):")
    lines.append("")
    lines.append("```text")
    lines.append("I(x, y) = 0.299 * R(x, y) + 0.587 * G(x, y) + 0.114 * B(x, y)")
    lines.append("```")
    lines.append("")
    lines.append("Адаптивный порог NICK:")
    lines.append("")
    lines.append("```text")
    lines.append("m(x, y)      = (1/|W|) * sum(I(i, j)),      (i, j) in W(x, y)")
    lines.append("sqmean(x, y) = (1/|W|) * sum(I(i, j)^2),    (i, j) in W(x, y)")
    lines.append("var(x, y)    = sqmean(x, y) - m(x, y)^2")
    lines.append("T(x, y)      = m(x, y) + k * sqrt(var(x, y) + sqmean(x, y))")
    lines.append("B(x, y)      = 255, если I(x, y) > T(x, y), иначе 0")
    lines.append("```")
    lines.append("")
    lines.append("### 1. Приведение полноцветного изображения к полутоновому")
    lines.append("")

    for case in cases:
        lines.append(f"#### 1.{case.case_no} Изображение {case.case_no}")
        lines.append(f"Источник: `{case.source_url}`")
        lines.append("")
        lines.append("| Исходное (RGB, PNG) | Полутоновое (BMP) |")
        lines.append("|:-------------------:|:-----------------:|")
        lines.append(f"| ![source](src/{case.source_name}) | ![gray](src/{case.gray_name}) |")
        lines.append("")

    lines.append("### 2. Бинаризация полутонового изображения методом NICK")
    lines.append("")
    for case in cases:
        lines.append(f"#### 2.{case.case_no} Изображение {case.case_no}")
        lines.append("")
        header = "| Полутоновое | " + " | ".join(f"NICK {w}x{w}" for w in WINDOW_SIZES) + " |"
        align = "|:-----------:|" + "|".join([":---------:" for _ in WINDOW_SIZES]) + "|"
        cells = [f"![gray](src/{case.gray_name})"] + [
            f"![w{w}](src/{case.binary_names[w]})" for w in WINDOW_SIZES
        ]
        row = "| " + " | ".join(cells) + " |"
        lines.append(header)
        lines.append(align)
        lines.append(row)
        lines.append("")

    lines.append("### Результаты выполнения")
    lines.append("")
    lines.append("| Изображение | Размер | Бинарные файлы |")
    lines.append("|:------------|-------:|:---------------|")
    for case in cases:
        size = f"{case.width}x{case.height}"
        binary_files = ", ".join(case.binary_names[w] for w in WINDOW_SIZES)
        lines.append(
            f"| №{case.case_no} (индекс {case.image_index}) | {size} | `{binary_files}` |"
        )
    lines.append("")
    lines.append("### Выводы")
    lines.append("")
    lines.append("1. Реализовано обесцвечивание RGB-изображений без библиотечной функции перевода в grayscale.")
    lines.append(
        "2. Для варианта 12 реализована адаптивная бинаризация NICK с окнами 3x3 и 25x25 без библиотечных функций бинаризации."
    )
    lines.append("3. В отчете показаны результаты каждой операции (до и после) на нескольких изображениях.")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_generated_files(RESULTS_DIR)
    cleanup_generated_files(SRC_DIR)

    image_paths = fetch_image_paths(ORIGIN, SAMPLE_ID)
    if not image_paths:
        raise RuntimeError("Список изображений пуст.")

    cases: list[CaseResult] = []
    for case_no, image_index in enumerate(IMAGE_INDICES, start=1):
        if image_index < 0 or image_index >= len(image_paths):
            raise IndexError(f"Индекс {image_index} выходит за пределы списка image_paths.")

        source_url = image_paths[image_index]
        source_rgb = download_image_rgb(source_url)
        gray = rgb_to_grayscale_weighted(source_rgb)

        source_name = f"img{case_no}_source.png"
        gray_name = f"img{case_no}_grayscale.bmp"
        binary_names: dict[int, str] = {}

        save_rgb(source_rgb, RESULTS_DIR / source_name)
        save_gray(gray, RESULTS_DIR / gray_name)
        save_rgb(source_rgb, SRC_DIR / source_name)
        save_gray(gray, SRC_DIR / gray_name)

        for window_size in WINDOW_SIZES:
            binary = nick_binarization(gray, window_size=window_size, k_value=K_PARAM)
            binary_name = f"img{case_no}_binary_nick_w{window_size}.bmp"
            binary_names[window_size] = binary_name

            save_gray(binary, RESULTS_DIR / binary_name)
            save_gray(binary, SRC_DIR / binary_name)

        height, width = gray.shape
        cases.append(
            CaseResult(
                case_no=case_no,
                image_index=image_index,
                source_url=source_url,
                width=width,
                height=height,
                source_name=source_name,
                gray_name=gray_name,
                binary_names=binary_names,
            )
        )

    write_report(cases)

    print("Лабораторная работа №2 выполнена.")
    print(f"Вариант: {VARIANT} (NICK)")
    print(f"Окна: {', '.join(f'{w}x{w}' for w in WINDOW_SIZES)}")
    print(f"Результаты: {RESULTS_DIR}")
    print(f"Файлы для отчета: {SRC_DIR}")
    print(f"Отчет: {REPORT_PATH}")


if __name__ == "__main__":
    main()

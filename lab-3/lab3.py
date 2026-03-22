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
METHOD_NAME = "Фильтр преобладающего оттенка"
WINDOW_SIZE = 3

# Несколько изображений (выполнение на нескольких входах по требованию).
IMAGE_INDICES = [4, 24]

DIFF_GRAY_VIS_SCALE = 4

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
    gray_filtered_name: str
    gray_diff_name: str
    gray_diff_vis_name: str
    mono_name: str
    mono_filtered_name: str
    mono_diff_name: str


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
    for file_path in directory.glob("case*"):
        if file_path.is_file():
            file_path.unlink()


def rgb_to_grayscale_weighted(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float64)
    gray = 0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2]
    return np.clip(gray, 0, 255).round().astype(np.uint8)


def to_monochrome_by_mean(gray: np.ndarray) -> np.ndarray:
    threshold = float(gray.mean())
    return np.where(gray > threshold, 255, 0).astype(np.uint8)


def predominant_shade_filter(gray: np.ndarray, window_size: int) -> np.ndarray:
    if window_size % 2 == 0 or window_size < 3:
        raise ValueError("Размер окна должен быть нечетным и >= 3.")

    radius = window_size // 2
    padded = np.pad(gray, ((radius, radius), (radius, radius)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (window_size, window_size))
    values = windows.reshape(gray.shape[0], gray.shape[1], window_size * window_size)

    # Фильтр преобладающего оттенка: берется мода по окну.
    sorted_values = np.sort(values, axis=-1)
    diff = np.diff(sorted_values, axis=-1)
    run_starts = np.concatenate(
        [
            np.ones((*sorted_values.shape[:2], 1), dtype=bool),
            diff != 0,
        ],
        axis=-1,
    )

    counts = np.ones_like(sorted_values, dtype=np.uint8)
    for i in range(sorted_values.shape[-1] - 2, -1, -1):
        same = sorted_values[..., i] == sorted_values[..., i + 1]
        counts[..., i] = np.where(same, counts[..., i + 1] + 1, 1)

    run_lengths = np.where(run_starts, counts, 0)
    best_len = run_lengths.max(axis=-1, keepdims=True)
    best_positions = run_lengths == best_len
    best_idx = best_positions.argmax(axis=-1)
    mode = np.take_along_axis(sorted_values, best_idx[..., None], axis=-1)[..., 0]
    return mode.astype(np.uint8)


def abs_difference(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    return np.abs(img_a.astype(np.int16) - img_b.astype(np.int16)).astype(np.uint8)


def xor_difference(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    return np.bitwise_xor(img_a, img_b).astype(np.uint8)


def write_report(cases: list[CaseResult]) -> None:
    lines: list[str] = []
    lines.append("# Лабораторная работа №3")
    lines.append("## Фильтрация изображений и морфологические операции")
    lines.append("")
    lines.append(f"### Вариант {VARIANT}: {METHOD_NAME}, окно {WINDOW_SIZE}x{WINDOW_SIZE}")
    lines.append("")
    lines.append("### Исходные данные")
    lines.append(f"- Количество изображений: {len(cases)}")
    lines.append("- Использованы полноцветные изображения, переведенные в полутон и монохром.")
    lines.append("- Разностные изображения:")
    lines.append("  - полутон: модуль разности `|I - F|` (дополнительно показана версия с усилением контраста);")
    lines.append("  - монохром: `XOR(I, F)`.")
    lines.append("")
    lines.append("### Формулы")
    lines.append("")
    lines.append("Перевод в полутон:")
    lines.append("")
    lines.append("```text")
    lines.append("I(x, y) = 0.299 * R(x, y) + 0.587 * G(x, y) + 0.114 * B(x, y)")
    lines.append("```")
    lines.append("")
    lines.append("Фильтр преобладающего оттенка (мода в окне W):")
    lines.append("")
    lines.append("```text")
    lines.append("F(x, y) = argmax_v count(v, W(x, y))")
    lines.append("```")
    lines.append("")
    lines.append("Разностные изображения:")
    lines.append("")
    lines.append("```text")
    lines.append("D_gray(x, y) = |I(x, y) - F(x, y)|")
    lines.append("D_mono(x, y) = I_mono(x, y) XOR F_mono(x, y)")
    lines.append("```")
    lines.append("")
    lines.append("### 1. Полутоновая обработка")
    lines.append("")
    for case in cases:
        lines.append(f"#### 1.{case.case_no} Изображение {case.case_no}")
        lines.append(f"Источник: `{case.source_url}`")
        lines.append("")
        lines.append("| Исходное RGB | Полутоновое | После фильтра | Разностное `|I-F|` | Разностное (x4) |")
        lines.append("|:------------:|:-----------:|:-------------:|:------------------:|:---------------:|")
        lines.append(
            f"| ![src](src/{case.source_name}) | ![gray](src/{case.gray_name}) | ![grayf](src/{case.gray_filtered_name}) | ![diff](src/{case.gray_diff_name}) | ![diffv](src/{case.gray_diff_vis_name}) |"
        )
        lines.append("")

    lines.append("### 2. Монохромная обработка")
    lines.append("")
    for case in cases:
        lines.append(f"#### 2.{case.case_no} Изображение {case.case_no}")
        lines.append("")
        lines.append("| Исходное монохромное | После фильтра | XOR-разность |")
        lines.append("|:--------------------:|:-------------:|:------------:|")
        lines.append(
            f"| ![mono](src/{case.mono_name}) | ![monof](src/{case.mono_filtered_name}) | ![xord](src/{case.mono_diff_name}) |"
        )
        lines.append("")

    lines.append("### Результаты выполнения")
    lines.append("")
    lines.append("| Изображение | Размер | Фильтр |")
    lines.append("|:------------|-------:|:-------|")
    for case in cases:
        size = f"{case.width}x{case.height}"
        lines.append(f"| №{case.case_no} (индекс {case.image_index}) | {size} | {METHOD_NAME}, окно {WINDOW_SIZE}x{WINDOW_SIZE} |")
    lines.append("")
    lines.append("### Выводы")
    lines.append("")
    lines.append("1. Реализован фильтр преобладающего оттенка (вариант 12) без библиотечных функций фильтрации.")
    lines.append("2. Получены отфильтрованные изображения в полутоне и монохроме.")
    lines.append("3. Сформированы разностные изображения: модуль разности для полутона и XOR для монохрома.")

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
        gray_filtered = predominant_shade_filter(gray, window_size=WINDOW_SIZE)
        gray_diff = abs_difference(gray, gray_filtered)
        gray_diff_vis = np.clip(
            gray_diff.astype(np.uint16) * DIFF_GRAY_VIS_SCALE, 0, 255
        ).astype(np.uint8)

        mono = to_monochrome_by_mean(gray)
        mono_filtered = predominant_shade_filter(mono, window_size=WINDOW_SIZE)
        mono_diff = xor_difference(mono, mono_filtered)

        source_name = f"case{case_no}_source.png"
        gray_name = f"case{case_no}_gray.bmp"
        gray_filtered_name = f"case{case_no}_gray_filtered_v{VARIANT}.bmp"
        gray_diff_name = f"case{case_no}_gray_diff_abs.bmp"
        gray_diff_vis_name = f"case{case_no}_gray_diff_abs_x{DIFF_GRAY_VIS_SCALE}.bmp"
        mono_name = f"case{case_no}_mono.bmp"
        mono_filtered_name = f"case{case_no}_mono_filtered_v{VARIANT}.bmp"
        mono_diff_name = f"case{case_no}_mono_diff_xor.bmp"

        for out_dir in (RESULTS_DIR, SRC_DIR):
            save_rgb(source_rgb, out_dir / source_name)
            save_gray(gray, out_dir / gray_name)
            save_gray(gray_filtered, out_dir / gray_filtered_name)
            save_gray(gray_diff, out_dir / gray_diff_name)
            save_gray(gray_diff_vis, out_dir / gray_diff_vis_name)
            save_gray(mono, out_dir / mono_name)
            save_gray(mono_filtered, out_dir / mono_filtered_name)
            save_gray(mono_diff, out_dir / mono_diff_name)

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
                gray_filtered_name=gray_filtered_name,
                gray_diff_name=gray_diff_name,
                gray_diff_vis_name=gray_diff_vis_name,
                mono_name=mono_name,
                mono_filtered_name=mono_filtered_name,
                mono_diff_name=mono_diff_name,
            )
        )

    write_report(cases)

    print("Лабораторная работа №3 выполнена.")
    print(f"Вариант: {VARIANT}")
    print(f"Метод: {METHOD_NAME}, окно {WINDOW_SIZE}x{WINDOW_SIZE}")
    print(f"Результаты: {RESULTS_DIR}")
    print(f"Файлы для отчета: {SRC_DIR}")
    print(f"Отчет: {REPORT_PATH}")


if __name__ == "__main__":
    main()

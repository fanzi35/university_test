import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import CHINESE_FONTS


XLSX_NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def ensure_dir(path: Path) -> None:
    """确保输出目录存在。"""
    path.mkdir(parents=True, exist_ok=True)


def setup_chinese_font() -> None:
    """配置 matplotlib 中文显示。"""
    plt.rcParams["font.sans-serif"] = CHINESE_FONTS
    plt.rcParams["axes.unicode_minus"] = False


def read_xlsx_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    """不依赖 openpyxl，按工作表名称读取 xlsx 为 DataFrame。"""
    with zipfile.ZipFile(path) as workbook:
        shared_strings = _read_shared_strings(workbook)
        sheet_paths = _read_sheet_paths(workbook)
        if sheet_name not in sheet_paths:
            raise ValueError(f"未找到工作表：{sheet_name}")
        rows = _read_sheet_rows(workbook, sheet_paths[sheet_name], shared_strings)

    if not rows:
        return pd.DataFrame()

    header = rows[0]
    data = rows[1:]
    return pd.DataFrame(data, columns=header)


def _read_shared_strings(workbook: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in workbook.namelist():
        return []

    root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
    strings = []
    for item in root.findall("a:si", XLSX_NS):
        text = "".join((node.text or "") for node in item.findall(".//a:t", XLSX_NS))
        strings.append(text)
    return strings


def _read_sheet_paths(workbook: zipfile.ZipFile) -> dict[str, str]:
    workbook_root = ET.fromstring(workbook.read("xl/workbook.xml"))
    rels_root = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
    rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels_root}

    paths = {}
    for sheet in workbook_root.findall("a:sheets/a:sheet", XLSX_NS):
        name = sheet.attrib["name"]
        rel_id = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        paths[name] = "xl/" + rel_map[rel_id]
    return paths


def _read_sheet_rows(
    workbook: zipfile.ZipFile,
    sheet_path: str,
    shared_strings: list[str],
) -> list[list[object]]:
    root = ET.fromstring(workbook.read(sheet_path))
    parsed_rows = []
    max_col = 0

    for row in root.findall("a:sheetData/a:row", XLSX_NS):
        values = {}
        for cell in row.findall("a:c", XLSX_NS):
            column_index = _column_index(cell.attrib.get("r", ""))
            max_col = max(max_col, column_index)
            values[column_index] = _cell_value(cell, shared_strings)
        parsed_rows.append(values)

    rows = []
    for values in parsed_rows:
        row_values = [values.get(col, None) for col in range(1, max_col + 1)]
        if any(value not in (None, "") for value in row_values):
            rows.append(row_values)
    return rows


def _column_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)\d+", cell_ref)
    if not match:
        return 0

    index = 0
    for char in match.group(1):
        index = index * 26 + ord(char) - ord("A") + 1
    return index


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> object:
    value = cell.find("a:v", XLSX_NS)
    if value is None:
        return None

    text = value.text or ""
    if cell.attrib.get("t") == "s":
        return shared_strings[int(text)]

    try:
        return float(text)
    except ValueError:
        return text

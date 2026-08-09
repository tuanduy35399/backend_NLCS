import json
import requests
import re
from pathlib import Path
from bs4 import BeautifulSoup


BASE_URL = "https://tuyensinh.ctu.edu.vn"

headers = {
    "User-Agent": "Mozilla/5.0"
}


def get_major_links():
    links = []

    for start in range(0, 130, 10):
        page_url = (
            f"{BASE_URL}/gioi-thieu-nganh.html"
            f"?limit=10&start={start}"
        )

        print(f"Đang đọc: {page_url}")

        res = requests.get(page_url, headers=headers, timeout=30)
        res.raise_for_status()

        soup = BeautifulSoup(res.text, "lxml")

        for a in soup.select("a[href]"):
            href = a["href"]

            if not href.endswith(".html"):
                continue

            if "gioi-thieu-nganh/" not in href:
                continue

            if href.startswith("http"):
                url = href
            else:
                url = BASE_URL + "/" + href.lstrip("/")

            if url not in links:
                links.append(url)

    return links


def scrape_major(url):
    res = requests.get(url, headers=headers, timeout=30)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "lxml")
    content = soup.select_one("section.article-content.clearfix")

    paragraphs = []

    if content:
        for tag in content.find_all(["video", "source"]):
            tag.decompose()

        for p in content.find_all("p"):
            txt = p.get_text(" ", strip=True)

            if txt:
                paragraphs.append(txt)

    text = "\n\n".join(paragraphs)

    match = re.search(
        r"(?:^|\n|[-•])\s*Tên ngành\s*:\s*([^\n]+)",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        title = match.group(1).strip()
    else:
        title = "Không xác định"

    return {
        "ten_nganh": title,
        "url": url,
        "noi_dung": text,
    }


def export_json(data):
    output_path = (
        Path(__file__).resolve().parents[2]
        / "old_rag"
        / "rag"
        / "app"
        / "google_model"
        / "ctu_majors.json"
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Đã lưu file tại: {output_path}")


if __name__ == "__main__":
    links = get_major_links()

    print("Tổng số ngành:", len(links))

    majors = []

    for i, link in enumerate(links):
        print(f"[{i + 1}/{len(links)}] {link}")

        try:
            major = scrape_major(link)
            majors.append(major)

        except Exception as e:
            print(f"Lỗi: {e}")

    export_json(majors)
    print("Đã tạo ctu_majors.json")
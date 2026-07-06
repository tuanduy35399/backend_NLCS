import json
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://tuyensinh.ctu.edu.vn"
LIST_URL = BASE_URL + "/gioi-thieu-nganh/"

headers = {
    "User-Agent": "Mozilla/5.0"
}


# -------------------------------
# Lấy tất cả link ngành
# -------------------------------
def get_major_links():
    links = []

    # Có 13 trang, mỗi trang 10 ngành
    for start in range(0, 130, 10):

        page_url = f"{BASE_URL}/gioi-thieu-nganh.html?limit=10&start={start}"
        print(f"Đang đọc: {page_url}")

        res = requests.get(page_url, headers=headers)
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


# -------------------------------
# Lấy nội dung ngành
# -------------------------------
def scrape_major(url):
    res = requests.get(url, headers=headers)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "lxml")

    # Tiêu đề
    title = soup.select_one("h1, h2")
    title = title.get_text(strip=True) if title else "Không có tiêu đề"

    # Nội dung bài viết
    content = soup.select_one("section.article-content.clearfix")

    paragraphs = []

    if content:

        # bỏ video
        for tag in content.find_all(["video", "source"]):
            tag.decompose()

        for p in content.find_all("p"):

            txt = p.get_text(" ", strip=True)

            if txt:
                paragraphs.append(txt)

    text = "\n\n".join(paragraphs)
    return {
        "ten_nganh": title,
        "url": url,
        "noi_dung": text
    }


# -------------------------------
# Xuất JSON
# -------------------------------
def export_json(data, filename="ctu_majors.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":

    links = get_major_links()

    print("Tổng số ngành:", len(links))

    majors = []

    for i, link in enumerate(links):
        print(f"[{i+1}/{len(links)}] {link}")

        try:
            major = scrape_major(link)
            majors.append(major)

        except Exception as e:
            print(f"Lỗi: {e}")

    export_json(majors)

    print("Đã tạo file ctu_majors.json")
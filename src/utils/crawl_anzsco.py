import requests
from bs4 import BeautifulSoup
import pandas as pd
import json

# -------------------------
# Core extraction function
# -------------------------
def extract_anzsco_info(url: str) -> dict:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Occupation name
    occupation_name = soup.find("h1")
    if not occupation_name:
        occupation_name = soup.find("title")
    occupation_name = occupation_name.get_text(strip=True) if occupation_name else None

    # Skill level
    skill_level = None
    for p in soup.find_all(["p", "li", "div"]):
        text = p.get_text(strip=True)
        if text.startswith("Skill Level") or text.startswith("Indicative Skill Level"):
            skill_level = text
            break

    # Tasks
    tasks = []
    task_header = soup.find(lambda tag: tag.name in ["p","strong","h3","h4"] 
                            and "Tasks Include" in tag.get_text())
    if task_header:
        ul = task_header.find_next("ul")
        if ul:
            tasks = [li.get_text(strip=True) for li in ul.find_all("li")]
        else:
            sibs = task_header.find_all_next("p", limit=6)
            for s in sibs:
                if s.get_text(strip=True):
                    tasks.append(s.get_text(strip=True))

    # Subcategories
    subcategories = []
    subcat_header = soup.find(lambda tag: tag.name in ["p","strong","h3","h4"] 
                              and "Subcategories" in tag.get_text())
    if subcat_header:
        for a in subcat_header.find_all_next("a", href=True):
            text = a.get_text(strip=True)
            if text and text[0].isdigit():
                subcategories.append({
                    "name": text,
                    "url": "https://www.abs.gov.au" + a["href"]
                })
            else:
                break

    return {
        "occupation_name": occupation_name,
        "skill_level": skill_level,
        "tasks": tasks,
        "subcategories": subcategories
    }

# -------------------------
# Recursive crawler
# -------------------------
def crawl_anzsco(url: str, depth: int = 0, max_depth: int = 10) -> dict:
    node = extract_anzsco_info(url)

    if depth < max_depth and node["subcategories"]:
        children = []
        for sub in node["subcategories"]:
            try:
                child = crawl_anzsco(sub["url"], depth + 1, max_depth)
                children.append(child)
            except Exception as e:
                print(f"Failed to crawl {sub['url']}: {e}")
        node["children"] = children
    else:
        node["children"] = []

    return node

# -------------------------
# Flatten tree into rows
# -------------------------
def flatten_tree(node, parent_path=None):
    if parent_path is None:
        parent_path = []
    
    current_path = parent_path + [node.get("occupation_name")]
    
    row = {
        "occupation_name": node.get("occupation_name"),
        "skill_level": node.get("skill_level"),
        "tasks": "; ".join(node.get("tasks", [])),
        "path": " > ".join([p for p in current_path if p])
    }
    
    rows = [row]
    
    for child in node.get("children", []):
        rows.extend(flatten_tree(child, current_path))
    
    return rows

# # -------------------------
# # Example usage
# # Crawl ALL major groups (1–8)
# # -------------------------
# base = "https://www.abs.gov.au/statistics/classifications/anzsco-australian-and-new-zealand-standard-classification-occupations/2022/browse-classification/"
# major_groups = [str(i) for i in range(1, 9)]
# start_urls = [base + g for g in major_groups]

# all_rows = []
# for url in start_urls:
#     print(f"Crawling {url} ...")
#     tree = crawl_anzsco(url, max_depth=6)  # adjust depth if needed
#     rows = flatten_tree(tree)
#     all_rows.extend(rows)

# # Convert to DataFrame
# df = pd.DataFrame(all_rows)

# # Save to CSV (optional)
# df.to_csv("anzsco_full_flat.csv", index=False, encoding="utf-8")

# print("✅ Completed crawl. DataFrame shape:", df.shape)
# print(df.head(10))

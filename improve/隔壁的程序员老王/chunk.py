from httpx import get


def read_data():
    with open("data.md", "r", encoding='utf-8') as f:
        return f.read()
    
def get_chunks() -> list[str]:
    content = read_data()
    chunks = content.split('\n\n')
    
    header = ""
    result = []
    for c in chunks:
        if c.startswith("#"):
            header += c
        else:
            result.append(f"{header}{c}")
            header = ""

    return result

if __name__ == "__main__":
    chunks = get_chunks()
    for c in chunks:
        print(c)
        print("----------")

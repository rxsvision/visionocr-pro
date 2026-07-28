"""文件存储管理"""
import shutil
from datetime import datetime
from pathlib import Path


class Storage:
    def __init__(self, data_dir: str):
        self.root = Path(data_dir)
        self.uploads = self.root / "uploads"
        self.results = self.root / "results"
        self.exports = self.root / "exports"
        for d in (self.uploads, self.results, self.exports):
            d.mkdir(parents=True, exist_ok=True)

    def save_upload(self, file_path: str | Path, category: str = "general") -> Path:
        """保存上传文件, 返回存储路径"""
        src = Path(file_path)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_dir = self.uploads / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{ts}_{src.name}"
        shutil.copy2(str(src), str(dest))
        return dest

    def save_result(self, content: str, name: str, ext: str = ".json") -> Path:
        """保存结果文件"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = self.results / f"{ts}_{name}{ext}"
        dest.write_text(content, encoding="utf-8")
        return dest

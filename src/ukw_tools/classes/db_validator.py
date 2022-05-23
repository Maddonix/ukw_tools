from pathlib import Path

from pydantic import BaseModel
from tqdm import tqdm

from .db import DbHandler


class DbValidator(BaseModel):
    db: DbHandler

    def get_missing_extracted_imgs(self):
        imgs = self.db.image.find({"is_extracted": True})
        i = 0
        failed = []
        for img in tqdm(imgs):
            if i == 10000:
                i = 0
                _ = self.db.image.find_one({})  # keep Db Alive
            i += 1
            path = Path(img["path"])
            if not path.exists():
                failed.append(img)

        return failed

# election-news

חדשות שלוחה 1 (קו 0794949476): איסוף כתבות וסקרים, המרה לאודיו, העלאה לימות ומשלוח וידאו במייל.
הועבר מהמאגר `SHMUEL-LAMED/07` עם ההיסטוריה המלאה.

## מבנה
- `election-news/` – האוסף, ה-Worker `election-news-ivr` והבדיקות (פירוט ב-`election-news/README.md`).
- `scripts/` – פריסה (`deploy-election-news.py`, `deploy-news-archive.py`), החלפת קו יעד, ובניית גשר הווידאו (`build-mail-video.py`).
- `live/mail/` – הבסיס לגשר הווידאו שמוזרק ל-Worker `yemot-mail` (הקוד של הקו עצמו במאגר `Email-management-line`).
- `.github/workflows/` – איסוף מתוזמן, פריסה והחלפת קו.

## דרוש בהגדרות המאגר
- הסוד `CLOUDFLARE_API_TOKEN` ב-Actions.
- ב-Worker `election-news-ivr`: `COLLECTOR_REPO=SHMUEL-LAMED/election-news` ו-`COLLECTOR_REPO_ID=1457981746` (אימות OIDC של הסורק, רק מענף `main`).

## בדיקות
```bash
python -m unittest discover -s election-news -p 'test_*.py'
node --test election-news/*.test.mjs
```

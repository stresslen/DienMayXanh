# DMX Review Crawler

Crawler dùng Playwright API request context có checkpoint để thu thập review sản phẩm công khai trên Điện Máy Xanh.
Giai đoạn này chỉ lưu dữ liệu thô; chưa làm sạch và chưa gán nhãn Gemini.

## Cài đặt

```powershell
python -m pip install -e .
python -m playwright install chromium
```

## Chạy

Lấy URL sản phẩm từ sitemap chính thức:

```powershell
python -m dmx_crawler seed --since-year 2020
```

Crawl đến khi có 10.000 review hoặc đã xử lý 2.000 sản phẩm:

```powershell
python -m dmx_crawler crawl --target-reviews 10000 --max-products 2000 --concurrency 3
```

Có thể seed và crawl trong một lệnh:

```powershell
python -m dmx_crawler run --since-year 2020 --target-reviews 10000
```

Khoanh một năm để lấy mẫu hoặc tiếp tục một giai đoạn cụ thể:

```powershell
python -m dmx_crawler seed --since-year 2023 --until-year 2023
```

Xem thống kê và xuất dữ liệu:

```powershell
python -m dmx_crawler stats
python -m dmx_crawler validate
python -m dmx_crawler export
```

Đưa các URL đã hết số lần retry trở lại queue sau khi mạng ổn định:

```powershell
python -m dmx_crawler retry-errors
```

Dữ liệu chuẩn nằm trong `data/raw/dmx_reviews.sqlite3`. Hai file xuất:

- `data/raw/export/reviews_raw.jsonl` (UTF-8)
- `data/raw/export/reviews_raw.csv` (UTF-8 BOM, mở trực tiếp bằng Excel)

## Đặc tính vận hành

- Playwright request context dùng chung cookie/session, không tải ảnh, quảng cáo hay tài nguyên render.
- Dùng sitemap công khai và endpoint phân trang review của chính website.
- Mặc định tối đa 3 worker và delay ngẫu nhiên 0,8–1,8 giây.
- Retry cho lỗi tạm thời, checkpoint SQLite, resume tự động.
- Lỗi mạng Playwright được log một dòng an toàn trên Windows và retry đến `--max-attempts`.
- `source_review_id` là khóa duy nhất nên chạy lại không tạo bản ghi trùng.
- Không vượt đăng nhập, CAPTCHA hay đường dẫn bị cấm bởi `robots.txt`.

Nếu website thay đổi HTML hoặc endpoint, chạy test và smoke crawl trước khi chạy dài:

```powershell
python -m pytest -q
python -m dmx_crawler --database data/smoke.sqlite3 run --since-year 2026 --sitemap-limit 1 --target-reviews 20 --max-products 20 --concurrency 1
```

Để seed lại sitemap và vét các URL còn pending, bị ngắt giữa chừng hoặc đã lỗi
hết số lần thử, dùng `run --retry-errors`. Không chạy đồng thời hai crawler trên
cùng một file SQLite:

```powershell
python -u -m dmx_crawler --database data/raw/dmx_reviews.sqlite3 run --since-year 0 --oldest-first --retry-errors --target-reviews 2147483647 --max-products 2147483647 --max-review-pages 10000 --max-attempts 10 --concurrency 32 --min-delay 0.1 --max-delay 0.25
```

## Pipeline gán nhãn mức độ hài lòng

Luồng mới sử dụng toàn bộ bình luận thật trong SQLite raw, giữ cách làm sạch,
che PII, loại trùng, retry và checkpoint của pipeline hiện có. Nhãn được lưu
riêng tại `data/processed/sentiment/sentiment.sqlite3`:

| Giá trị nhãn | Tên tiếng Việt | Quy tắc |
| --- | --- | --- |
| `satisfied` | Hài lòng | Khen, hài lòng, khuyến nghị mua, đánh giá tích cực. |
| `dissatisfied` | Không hài lòng | Chê, thất vọng, lỗi ảnh hưởng sử dụng, yêu cầu đổi/trả vì lỗi. |
| `neutral` | Trung tính | Hỏi thông tin, mô tả thuần túy, chưa đánh giá, không rõ cảm xúc. |

Với câu khen/chê lẫn nhau, ưu tiên kết luận rõ và trải nghiệm chính; lỗi nghiêm
trọng/yêu cầu đổi trả có trọng số cao hơn lời khen phụ. Câu hỏi thông tin thông
thường là trung tính; câu hỏi kèm lỗi đang gặp là không hài lòng. Model chỉ nhận
văn bản đã làm sạch, không nhận số sao. Nhãn độc tính cũ không được chuyển thành
nhãn hài lòng. Prompt nằm trong `src/dmx_crawler/sentiment.py`.

Chuẩn bị workspace một lần:

```powershell
python -m dmx_crawler --database data/raw/dmx_reviews.sqlite3 prepare-sentiment
```

Thiết lập khóa Gemini trong shell, rồi gán nhãn. Mỗi request chứa tối đa **100**
bình luận unique; batch cuối chứa phần dư. Các bản trùng dùng lại nhãn để tiết
kiệm request. Muốn thử một request trước, thêm `--max-batches 1`:

```powershell
$env:GEMINI_API_KEY = "YOUR_API_KEY"
python -u -m dmx_crawler label-sentiment --batch-size 100
python -m dmx_crawler sentiment-stats
python -m dmx_crawler export-sentiment
```

Nếu dừng giữa chừng, chạy lại `label-sentiment --batch-size 100` để tiếp tục.
Không chạy lại `prepare-sentiment --force` khi đã có nhãn cần giữ. Khi có review
mới, chạy `sync-sentiment` trước khi tiếp tục gán nhãn.

Kết quả xuất nằm trong `data/processed/sentiment/export`:

- `reviews_clean.jsonl`: toàn bộ bình luận và trạng thái xử lý.
- `reviews_labeled.csv` / `reviews_labeled.jsonl`: các bình luận đã có nhãn.
- `training_sentiment_unique.csv`: các bình luận unique đã có nhãn để huấn luyện.
- `sentiment_stats.json`: thống kê tiến độ và nhãn.
- `sentiment_metadata.json`: tên nhãn tiếng Việt, phiên bản prompt và cấu hình.

Chỉ thống kê có `pending_unique=0` và `failed_unique=0` mới cho biết toàn bộ câu
unique đủ điều kiện đã được gán nhãn. Nội dung không có chữ được đánh dấu không
đủ điều kiện, không tự động gán trung tính.

## Chuẩn hóa và đóng gói DMX-Sentiment

Sau khi gán nhãn xong và dừng labeler, tạo bản dữ liệu có audit và ZIP:

```powershell
python -m pip install -e ".[dataset]"
python -m dmx_crawler curate-sentiment
```

Quy trình sao lưu database, dùng sao để phát hiện nhãn đáng nghi, chỉ sửa trực
tiếp những câu có bằng chứng nội dung rõ; không ánh xạ cứng số sao thành nhãn.
Các câu chưa chắc được lưu trong `audit/excluded.csv` để rà soát tiếp, không đi
vào tập huấn luyện chính. Có log trước/sau cho mỗi lần sửa.

Loại trùng chính xác và gần trùng, loại nhóm trùng có nhãn mâu thuẫn; chia
train/validation/test khoảng 80/10/10, phân tầng nhãn và giữ các sản phẩm tách
biệt giữa các tập. Gói chứa CSV, JSONL, Parquet, dataset card, mapping nhãn,
schema, checksum và báo cáo chất lượng. Sao chỉ nằm trong phần audit, không nằm
trong đầu vào huấn luyện. Nhãn vẫn là nhãn tự động (silver), chưa phải gold.

Mặc định tạo `data/processed/sentiment/DMX-Sentiment-v1.zip`. Nếu đã có bản
phát hành, chọn `--output-dir` mới để tránh ghi đè kết quả.

## Pipeline gán nhãn toxic / non_toxic (cũ)

Chuẩn bị dữ liệu từ SQLite raw. Lệnh này làm sạch tối thiểu, che PII, đánh dấu
dữ liệu ít thông tin và gom các nội dung trùng chính xác. Không sửa dữ liệu raw:

```powershell
python -m dmx_crawler --database data/raw/dmx_reviews.sqlite3 prepare-toxicity
```

API key chỉ được đọc từ biến môi trường, không truyền trên command line và không
ghi vào log. Chạy thử đúng một request trước. Mặc định là 100 review/request;
có thể tăng tối đa lên 1.000 bằng `--batch-size 1000`:

```powershell
$env:GEMINI_API_KEY = "YOUR_NEW_API_KEY"
python -m dmx_crawler label-toxicity --batch-size 1000 --max-batches 1
python -m dmx_crawler toxicity-stats
```

Sau khi kiểm tra batch thử, tiếp tục toàn bộ từ checkpoint:

```powershell
python -m dmx_crawler label-toxicity --batch-size 1000
```

Batch cuối chứa phần dư nếu số review unique không chia hết cho batch size. Export dữ
liệu sạch, toàn bộ nhãn và tập huấn luyện đã bỏ exact duplicate:

```powershell
python -m dmx_crawler export-toxicity
```

## Bộ dữ liệu DMX-Toxicity10K

Tạo bộ dữ liệu 10.000 câu gồm 1.000 câu toxic và 9.000 câu non-toxic. Pipeline
giữ các câu toxic thật đã gán nhãn, sinh có kiểm soát phần toxic còn thiếu, loại
câu trùng/gần trùng, lấy mẫu non-toxic thật phân tầng theo `Topic`, rồi chia
train/valid/test thành 7.000/2.000/1.000 câu:

```powershell
python -m dmx_crawler export-dmx-toxicity-10k
```

Kết quả nằm trong `data/processed/toxicity/dmx_toxicity_10k`. Đây là bộ dữ liệu
do dự án tự thu thập, gán nhãn và tăng cường. Các CSV dùng thứ tự cột
`Comment,Constructiveness,Toxicity,Title,Topic`; `Toxicity` dùng 1 cho toxic và
0 cho non-toxic. Vì pipeline hiện chỉ gán nhãn độc hại,
`Constructiveness` được để trống thay vì tạo nhãn giả. Dùng `--seed` nếu muốn
tạo một mẫu xác định khác. File `provenance.csv` ghi nguồn thật/synthetic cho
từng dòng; không coi dữ liệu synthetic là bình luận khách hàng quan sát được.

Các file kết quả nằm trong `data/processed/toxicity/export`. Không chạy lại
`prepare-toxicity` trên workspace đã có nhãn; tùy chọn `--force` sẽ chủ động thay
thế workspace và xóa toàn bộ checkpoint/nhãn cũ.

Khi crawler có thêm review mới, đồng bộ tăng dần để giữ nguyên toàn bộ nhãn cũ.
Review mới trùng nội dung sẽ tự dùng lại nhãn; chỉ nội dung unique chưa có nhãn
được đưa vào queue Gemini:

```powershell
python -m dmx_crawler --database data/raw/dmx_reviews.sqlite3 sync-toxicity
python -m dmx_crawler toxicity-stats
python -m dmx_crawler label-toxicity --batch-size 100
python -m dmx_crawler export-toxicity
```

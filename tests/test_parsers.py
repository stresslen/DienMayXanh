from dmx_crawler.models import Product
from dmx_crawler.parsers import (
    category_from_url,
    parse_product,
    parse_product_sitemap,
    parse_reviews,
    parse_sitemap_index,
)
from dmx_crawler.crawler import short_error, sitemap_sort_key


def test_parse_sitemap_index_and_product_sitemap():
    index = """<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <sitemap><loc>https://example.test/sitemap-1</loc></sitemap>
    </sitemapindex>"""
    sitemap = """<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://example.test/tu-lanh/demo</loc><lastmod>2026-01-01</lastmod></url>
    </urlset>"""
    assert parse_sitemap_index(index) == ["https://example.test/sitemap-1"]
    assert parse_product_sitemap(sitemap) == [
        ("https://example.test/tu-lanh/demo", "2026-01-01")
    ]


def test_parse_product():
    html = """
    <html><body>
      <div class="wrap_rating" data-objectid="363971" data-objecttype="2"
           data-siteid="2" data-title="Máy lạnh Demo"></div>
      <span class="point-alltimerate">1,2k đánh giá</span>
    </body></html>
    """
    product = parse_product(html, "https://www.dienmayxanh.com/may-lanh/demo")
    assert product is not None
    assert product.product_id == "363971"
    assert product.title == "Máy lạnh Demo"
    assert product.category == "may-lanh"
    assert product.reported_review_count == 1200


def test_parse_reviews():
    html = """
    <ul class="comment-list">
      <li id="r-59951580" class="par">
        <div class="confirm-buy">Đã mua tại ĐMX</div>
        <div class="cmt-top-star">
          <i class="iconcmt-starbuy"></i><i class="iconcmt-starbuy"></i>
          <i class="iconcmt-starbuy"></i><i class="iconcmt-starbuy"></i>
          <i class="iconcmt-unstarbuy"></i>
        </div>
        <div class="cmt-content"><p class="cmt-txt">Sản phẩm dùng tốt.</p></div>
        <div class="cmt-command">
          <a class="cmtl" data-like="12">Hữu ích</a>
          <span class="cmtd dot-line">Đã dùng khoảng 3 ngày</span>
        </div>
      </li>
    </ul>
    """
    product = Product("363971", "https://example.test/may-lanh/demo", "Demo", "may-lanh")
    rows = parse_reviews(html, product, 1)
    assert len(rows) == 1
    assert rows[0].source_review_id == "59951580"
    assert rows[0].rating == 4
    assert rows[0].text_raw == "Sản phẩm dùng tốt."
    assert rows[0].verified_purchase is True
    assert rows[0].helpful_count == 12


def test_category_from_url():
    assert category_from_url("https://www.dienmayxanh.com/tu-lanh/demo") == "tu-lanh"


def test_sitemap_sort_key_is_numeric():
    assert sitemap_sort_key(
        "https://example.test/sitemap-product-2026-10?page=2"
    ) > sitemap_sort_key("https://example.test/sitemap-product-2026-9?page=15")


def test_short_error_drops_multiline_playwright_call_log():
    error = Exception("APIRequestContext.get: socket hang up\nCall log:\n  → GET https://example.test")
    assert short_error(error) == "APIRequestContext.get: socket hang up"

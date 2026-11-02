from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Sequence

from .cleaning import clean_review_text


TARGETS = (
    "nhân viên bán hàng",
    "nhân viên tư vấn",
    "nhân viên chăm sóc khách hàng",
    "tổng đài viên",
    "kỹ thuật viên",
    "thợ lắp đặt",
    "đội giao hàng",
    "bên bảo hành",
    "quản lý cửa hàng",
    "bên vận chuyển",
    "cửa hàng này",
    "hãng này",
)

TARGETS_BY_SCENARIO = {
    "sales": (
        "nhân viên bán hàng",
        "nhân viên tư vấn",
        "quản lý cửa hàng",
        "cửa hàng này",
        "hãng này",
    ),
    "support": (
        "nhân viên chăm sóc khách hàng",
        "tổng đài viên",
        "bên bảo hành",
        "quản lý cửa hàng",
        "cửa hàng này",
        "hãng này",
    ),
    "technical": (
        "kỹ thuật viên",
        "thợ lắp đặt",
        "bên bảo hành",
        "hãng này",
    ),
    "delivery": (
        "đội giao hàng",
        "bên vận chuyển",
        "cửa hàng này",
    ),
    "general": TARGETS,
}

SCENARIOS = (
    ("support", "hẹn bảo hành hết lần này tới lần khác mà vẫn không tới"),
    ("sales", "tư vấn một kiểu rồi lúc đổi trả lại nói kiểu khác"),
    ("support", "nhận đủ tiền xong thì mặc kệ khi khách báo lỗi"),
    ("delivery", "giao sai mẫu nhưng cứ đổ lỗi ngược cho khách"),
    ("technical", "lắp đặt ẩu rồi không chịu quay lại xử lý"),
    ("general", "nói chuyện với khách bằng giọng thách thức"),
    ("support", "đùn đẩy trách nhiệm giữa cửa hàng và hãng"),
    ("delivery", "giao hàng trầy xước nhưng nhất quyết không cho đổi"),
    ("sales", "báo phí phát sinh sau khi khách đã thanh toán"),
    ("support", "để khách chờ cả tuần mà không có một lời giải thích"),
    ("sales", "tự ý thêm phụ kiện vào hóa đơn mà không hỏi"),
    ("technical", "kiểm tra qua loa rồi kết luận lỗi do người dùng"),
    ("support", "hứa gọi lại nhưng im luôn nhiều ngày"),
    ("delivery", "mang hàng lỗi tới rồi vội đi không cho kiểm tra"),
    ("support", "né tránh mỗi khi khách hỏi tiến độ sửa chữa"),
    ("support", "giữ máy bảo hành quá lâu mà không báo tình trạng"),
    ("delivery", "giao thiếu phụ kiện rồi bảo khách tự chịu"),
    ("sales", "ghi sai thông tin đơn hàng nhưng không chịu sửa"),
    ("support", "bắt khách tự chở món đồ cồng kềnh đi bảo hành"),
    ("support", "không đọc nội dung khiếu nại mà trả lời máy móc"),
    ("technical", "báo đã xử lý xong trong khi lỗi vẫn còn nguyên"),
    ("delivery", "hủy lịch giao sát giờ nhưng không hề xin lỗi"),
    ("delivery", "ép khách nhận hàng dù thùng đã móp và rách"),
    ("general", "cãi tay đôi thay vì giải quyết lỗi sản phẩm"),
    ("sales", "quảng cáo đổi mới nhưng khi có lỗi lại tìm cách từ chối"),
    ("technical", "chuyển máy về mà không lắp lại cho khách"),
    ("support", "thu tiền nhanh nhưng hoàn tiền thì bắt chờ vô thời hạn"),
    ("delivery", "đọc sai địa chỉ rồi quay sang trách khách"),
    ("support", "làm thất lạc phiếu tiếp nhận bảo hành"),
    ("support", "liên tục ngắt máy khi khách đang phản ánh"),
)

ATTACKS = (
    ("insult", "{target} đúng là quá ngu ngốc"),
    ("insult", "{target} ăn nói mất dạy"),
    ("insult", "{target} làm việc như không có não"),
    ("insult", "{target} vừa láo vừa thiếu hiểu biết"),
    ("insult", "{target} cư xử vô học thật sự"),
    ("insult", "{target} chỉ giỏi nói dối khách"),
    ("degrading", "cả đám {target} là một lũ vô dụng"),
    ("degrading", "{target} làm việc như một đám hề"),
    ("degrading", "{target} không đáng để khách tôn trọng"),
    ("degrading", "cái kiểu của {target} chỉ đáng bị khinh"),
    ("degrading", "{target} đúng là hết thuốc chữa"),
    ("degrading", "{target} chỉ biết phá việc của khách"),
    ("profanity", "{target} làm ăn như quần què"),
    ("profanity", "{target} làm ăn như cứt"),
    ("profanity", "{target} phục vụ như cái đéo gì vậy"),
    ("profanity", "{target} nói chuyện nghe ngứa cả tai"),
    ("profanity", "{target} làm khách bực vãi"),
    ("profanity", "đéo hiểu {target} làm việc kiểu gì"),
    ("fraud_accusation", "{target} đúng là một lũ lừa đảo"),
    ("fraud_accusation", "{target} chỉ chực moi tiền của khách"),
    ("fraud_accusation", "{target} xử lý kiểu ăn cướp"),
    ("fraud_accusation", "{target} làm ăn gian dối và trơ trẽn"),
    ("fraud_accusation", "{target} cố tình giăng bẫy khách hàng"),
    ("fraud_accusation", "{target} nói toàn lời bố láo để bán được hàng"),
    ("hostility", "{target} biến khỏi ngành dịch vụ đi"),
    ("hostility", "{target} nên dẹp tiệm cho rồi"),
    ("hostility", "mong {target} sớm bị khách quay lưng"),
    ("hostility", "{target} đừng xuất hiện trước mặt khách nữa"),
    ("hostility", "{target} cút đi cho đỡ làm phiền người khác"),
    ("hostility", "không ai nên phải chịu đựng kiểu {target} như vậy"),
)

OPENINGS = (
    "Thật sự không thể chấp nhận nổi",
    "Bực mình từ sáng tới giờ",
    "Nói thật là quá thất vọng",
    "Trải nghiệm lần này quá tệ",
    "Mua hàng nhiều nơi chưa gặp kiểu này",
    "Tôi đã cố nói chuyện rất bình tĩnh",
    "Không muốn nặng lời mà vẫn phải nói",
    "Dịch vụ gì mà kinh khủng",
    "Khách bỏ tiền ra chứ đâu có đi xin",
    "Một lần là quá đủ",
    "Càng nghĩ càng tức",
    "Đây là lần cuối tôi mua ở đây",
    "Chờ mãi rồi nhận lại cách xử lý này",
    "Tưởng nơi lớn thì làm việc tử tế",
    "Ban đầu nói nghe hay lắm",
    "Không hiểu nổi cách phục vụ này",
)

CLOSINGS = (
    "mọi người nên cân nhắc trước khi mua",
    "tôi sẽ không quay lại lần nào nữa",
    "né chỗ này ra cho đỡ rước bực vào người",
    "yêu cầu quản lý xem lại ngay",
    "đừng để khách khác gặp cảnh tương tự",
    "mất tiền mà còn mua thêm cục tức",
    "quá đủ cho một lần mua hàng",
    "khuyên mọi người tránh xa",
    "dịch vụ thế này thì mất khách là đúng",
    "không bao giờ giới thiệu cho người quen",
    "cần chấn chỉnh cách làm việc ngay",
    "thật sự phí thời gian của khách",
    "đề nghị xử lý người phụ trách ca này",
    "đừng quảng cáo hay khi phục vụ tệ như vậy",
    "từ nay xin chừa",
    "đọc đánh giá này rồi hãy quyết định",
)

TEMPLATES = (
    "{opening}. {scenario}. {attack}. {closing}.",
    "{scenario}; {attack}. {closing}.",
    "Mua {product} xong mới thấy {scenario}. {attack}. {closing}.",
    "Với {product}, tôi gặp cảnh {scenario}. {attack}. {closing}.",
    "{opening}: {scenario}, vậy mà {attack_lower}. {closing}.",
    "{target_cap} {scenario}. Nói thẳng ra, {attack_lower}. {closing}.",
    "Ai định mua {product} thì biết là {scenario}. {attack}. {closing}.",
    "Tôi phản ánh chuyện {scenario}, nhưng {attack_lower}. {closing}.",
    "{scenario}. Không vòng vo nữa: {attack}. {closing}.",
    "Đơn {product} của tôi bị xử lý kiểu {scenario}. {attack}. {closing}.",
    "{opening}. Riêng vụ {scenario} đã đủ bực, còn {attack_lower}. {closing}.",
    "Chuyện là {scenario}; cuối cùng chỉ thấy {attack_lower}. {closing}.",
    "{attack}. {closing}.",
    "{scenario}. {attack}.",
    "{attack}. Quá bực.",
    "{target_cap} {scenario}; {attack_lower}.",
)

STYLES = (
    "standard",
    "standard",
    "lowercase",
    "chat",
    "no_accents",
    "emphatic",
    "compact",
    "censored",
)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _stable_number(seed: str, index: int, attempt: int, slot: str) -> int:
    value = f"{seed}:{index}:{attempt}:{slot}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _pick(items: Sequence, seed: str, index: int, attempt: int, slot: str):
    return items[_stable_number(seed, index, attempt, slot) % len(items)]


def _without_accents(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )


def _apply_style(text: str, style: str) -> str:
    if style == "lowercase":
        return text.lower()
    if style == "chat":
        replacements = {
            "không": "ko",
            "Không": "Ko",
            "được": "dc",
            "nhân viên": "nv",
            "bảo hành": "bh",
            "mọi người": "mn",
            "khách hàng": "kh",
            "vậy": "zậy",
        }
        for source, replacement in replacements.items():
            text = text.replace(source, replacement)
        return text
    if style == "no_accents":
        return _without_accents(text).lower()
    if style == "emphatic":
        first, separator, rest = text.partition(".")
        return first.upper() + ("!!!" + rest if separator else "!!!")
    if style == "compact":
        return text.replace(". ", " ").replace("; ", ", ")
    if style == "censored":
        replacements = {
            "mất dạy": "m*t d*y",
            "đéo": "đ*o",
            "cứt": "c*t",
            "quần què": "q**n q**",
            "cút": "c*t",
        }
        for source, replacement in replacements.items():
            text = text.replace(source, replacement)
        return text
    return text


def _token_set(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_RE.findall(text.casefold()))


def _too_similar(tokens: frozenset[str], prior: Sequence[frozenset[str]]) -> bool:
    if not tokens:
        return True
    for other in prior:
        union = len(tokens | other)
        if union and len(tokens & other) / union >= 0.88:
            return True
    return False


def _short_product(title: str) -> str:
    cleaned = clean_review_text(title)
    return cleaned if len(cleaned) <= 72 else cleaned[:69].rstrip() + "..."


def generate_synthetic_toxic(
    contexts: Sequence[dict],
    count: int,
    seed: str,
    forbidden_texts: Sequence[str] = (),
) -> list[dict]:
    """Generate deterministic, traceable toxic reviews grounded in real products."""

    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return []
    if not contexts:
        raise ValueError("at least one product context is required")

    ordered_contexts = sorted(
        contexts,
        key=lambda row: hashlib.sha256(
            f"{seed}:context:{row['source_review_id']}".encode("utf-8")
        ).hexdigest(),
    )
    used_normalized = {clean_review_text(text).casefold() for text in forbidden_texts}
    token_sets = [_token_set(text) for text in forbidden_texts]
    generated: list[dict] = []
    attempt = 0
    max_attempts = max(10_000, count * 100)
    while len(generated) < count and attempt < max_attempts:
        index = len(generated)
        context_index = _stable_number(seed, index, attempt, "context") % len(
            ordered_contexts
        )
        context = ordered_contexts[context_index]
        scenario_group, scenario = _pick(
            SCENARIOS, seed, index, attempt, "scenario"
        )
        target = _pick(
            TARGETS_BY_SCENARIO[scenario_group],
            seed,
            index,
            attempt,
            "target",
        )
        toxic_type, attack_template = _pick(
            ATTACKS, seed, index, attempt, "attack"
        )
        attack = attack_template.format(target=target)
        opening = _pick(OPENINGS, seed, index, attempt, "opening")
        closing = _pick(CLOSINGS, seed, index, attempt, "closing")
        template = _pick(TEMPLATES, seed, index, attempt, "template")
        style = STYLES[index % len(STYLES)]
        text = template.format(
            opening=opening,
            scenario=scenario,
            attack=attack,
            attack_lower=attack[:1].lower() + attack[1:],
            closing=closing,
            target_cap=target[:1].upper() + target[1:],
            product=_short_product(str(context["product_title"])),
        )
        text = clean_review_text(_apply_style(text, style))
        normalized = text.casefold()
        tokens = _token_set(text)
        if (
            6 <= len(tokens) <= 90
            and normalized not in used_normalized
            and not _too_similar(tokens, token_sets)
        ):
            source_id = "synthetic-toxic-" + hashlib.sha256(
                f"{seed}:{index}:{text}".encode("utf-8")
            ).hexdigest()[:20]
            generated.append(
                {
                    "source_review_id": source_id,
                    "product_title": str(context["product_title"]),
                    "category": str(context["category"]),
                    "text_clean": text,
                    "label": "toxic",
                    "_origin": "synthetic_controlled",
                    "_toxic_type": toxic_type,
                    "_style": style,
                }
            )
            used_normalized.add(normalized)
            token_sets.append(tokens)
        attempt += 1

    if len(generated) != count:
        raise RuntimeError(
            f"Could only generate {len(generated)} of {count} diverse toxic rows"
        )
    return generated


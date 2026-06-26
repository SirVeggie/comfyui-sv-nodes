from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from comfy_api.latest import io

PACK_DIR = Path(__file__).resolve().parent
CREDENTIALS_PATH = PACK_DIR / "danbooru_credentials.json"
EXAMPLE_CREDENTIALS_PATH = PACK_DIR / "danbooru_credentials.example.json"

BASE_URL = "https://danbooru.donmai.us"
REQUEST_TIMEOUT = 30
MAX_DOWNLOAD_ATTEMPTS = 20
SEARCH_BATCH_SIZE = 20


def _credentials_file() -> Path | None:
    if CREDENTIALS_PATH.is_file():
        return CREDENTIALS_PATH
    return None


def load_credentials() -> dict[str, str]:
    path = _credentials_file()
    if path is None:
        return {"username": "", "api_key": "", "user_agent": ""}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid credentials file format in {path}")
    return {
        "username": str(data.get("username", "")),
        "api_key": str(data.get("api_key", "")),
        "user_agent": str(data.get("user_agent", "")),
    }


def get_credentials() -> tuple[str, str]:
    creds = load_credentials()
    return creds["username"].strip(), creds["api_key"].strip()


def resolve_user_agent() -> str | None:
    configured = load_credentials().get("user_agent", "").strip()
    return configured or None


def _request_headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    user_agent = resolve_user_agent()
    if user_agent:
        headers["User-Agent"] = user_agent
    return headers


def build_tags_query(
    user_tags: str = "",
    *,
    randomize: bool = False,
    rating: str | None = None,
) -> str:
    parts: list[str] = []
    if user_tags.strip():
        parts.append(user_tags.strip())
    if rating:
        parts.append(f"rating:{rating}")
    if randomize:
        parts.append("order:random")
    if not parts:
        raise ValueError("At least one search tag is required")
    return " ".join(parts)


def build_random_tags(rating: str = "s") -> str:
    return f"rating:{rating}"


def resolve_post_url(url: str | None, base_url: str = BASE_URL) -> str:
    if not url:
        return ""
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("/"):
        return base_url.rstrip("/") + url
    return url


def is_post_downloadable(post: dict) -> bool:
    file_url = post.get("file_url")
    large_file_url = post.get("large_file_url")
    return bool(file_url) or bool(large_file_url)


def pick_image_url(post: dict) -> str:
    extension = str(post.get("file_ext") or post.get("extension") or "").lower()
    file_url = post.get("file_url")
    large_file_url = post.get("large_file_url")
    preview_file_url = post.get("preview_file_url")

    if extension == "zip" and large_file_url:
        return resolve_post_url(large_file_url)
    if file_url:
        return resolve_post_url(file_url)
    if large_file_url:
        return resolve_post_url(large_file_url)
    if preview_file_url:
        return resolve_post_url(preview_file_url)
    raise ValueError(f"Post {post.get('id')!r} has no downloadable image URL")


def _build_opener(username: str, api_key: str) -> urllib.request.OpenerDirector:
    if username and api_key:
        password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, BASE_URL, username, api_key)
        return urllib.request.build_opener(urllib.request.HTTPBasicAuthHandler(password_mgr))
    return urllib.request.build_opener()


def _cloudflare_hint(body: str) -> str:
    if "Just a moment" in body:
        return (
            " Danbooru returned a Cloudflare challenge page. "
            "Clear user_agent in danbooru_credentials.json if it is set."
        )
    return ""


def _api_request(path: str, params: dict[str, str], username: str = "", api_key: str = "") -> object:
    filtered = {key: value for key, value in params.items() if value}
    query = urllib.parse.urlencode(filtered)
    url = f"{BASE_URL}{path}"
    if query:
        url = f"{url}?{query}"

    req = urllib.request.Request(url, headers=_request_headers())
    opener = _build_opener(username, api_key)
    try:
        with opener.open(req, timeout=REQUEST_TIMEOUT) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        hint = _cloudflare_hint(body)
        raise RuntimeError(f"Danbooru API error {exc.code}: {body[:300]}.{hint}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Danbooru request failed: {exc.reason}") from exc


def _as_post_list(data: object) -> list[dict]:
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [post for post in data if isinstance(post, dict)]
    raise RuntimeError("Unexpected Danbooru response format")


def _first_downloadable_post(posts: list[dict], *, context: str) -> dict:
    for post in posts:
        if is_post_downloadable(post):
            return post
    raise ValueError(
        f"No downloadable Danbooru posts found for {context}. "
        "Posts may require a Gold account or are otherwise restricted."
    )


def _fetch_random_downloadable(
    tags: str,
    username: str,
    api_key: str,
    *,
    context: str,
) -> dict:
    last_post_id: object = None
    for _ in range(MAX_DOWNLOAD_ATTEMPTS):
        post = _as_post_list(
            _api_request("/posts/random.json", {"tags": tags}, username, api_key)
        )[0]
        last_post_id = post.get("id")
        if is_post_downloadable(post):
            return post
    raise ValueError(
        f"No downloadable Danbooru posts found after {MAX_DOWNLOAD_ATTEMPTS} attempts "
        f"for {context} (last post id: {last_post_id!r})."
    )


def fetch_post(tags: str, username: str = "", api_key: str = "") -> dict:
    data = _api_request(
        "/posts.json",
        {"limit": str(SEARCH_BATCH_SIZE), "tags": tags},
        username,
        api_key,
    )
    posts = _as_post_list(data)
    if not posts:
        raise ValueError(f"No Danbooru posts matched tags: {tags!r}")
    return _first_downloadable_post(posts, context=f"tags: {tags!r}")


def fetch_random_post(rating: str = "s", username: str = "", api_key: str = "") -> dict:
    tags = build_random_tags(rating)
    return _fetch_random_downloadable(
        tags,
        username,
        api_key,
        context=f"random rating:{rating}",
    )


def fetch_post_by_tags(
    tags: str,
    *,
    randomize: bool = True,
    rating: str | None = None,
    username: str = "",
    api_key: str = "",
) -> dict:
    query = build_tags_query(tags, randomize=False, rating=rating)
    if randomize:
        return _fetch_random_downloadable(
            query,
            username,
            api_key,
            context=f"tags: {query!r}",
        )
    return fetch_post(query, username=username, api_key=api_key)


def download_image_tensor(url: str, username: str = "", api_key: str = "") -> torch.Tensor:
    headers = _request_headers()
    req = urllib.request.Request(url, headers=headers)
    opener = _build_opener(username, api_key)
    try:
        with opener.open(req, timeout=REQUEST_TIMEOUT) as response:
            data = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Failed to download image ({exc.code}): {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download image: {exc.reason}") from exc

    img = Image.open(BytesIO(data)).convert("RGB")
    return torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)[None,]


def post_tags(post: dict) -> str:
    tag_string = post.get("tag_string")
    if isinstance(tag_string, str) and tag_string.strip():
        return tag_string.strip()
    general = post.get("tag_string_general")
    if isinstance(general, str) and general.strip():
        return general.strip()
    return ""


def post_page_url(post: dict) -> str:
    post_id = post.get("id")
    if post_id is None:
        return ""
    return f"{BASE_URL}/posts/{post_id}"


def fetch_danbooru_image(
    tags: str,
    username: str = "",
    api_key: str = "",
) -> tuple[torch.Tensor, str, str, str]:
    post = fetch_post(tags, username=username, api_key=api_key)
    image_url = pick_image_url(post)
    tensor = download_image_tensor(image_url, username=username, api_key=api_key)
    tags_out = post_tags(post)
    post_id = str(post.get("id", ""))
    return tensor, tags_out, post_id, post_page_url(post)


class DanbooruRandomImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SV-DanbooruRandomImage",
            display_name="Danbooru Random Image",
            category="SV Nodes/Input",
            inputs=[
                io.Int.Input("seed", default=0, min=0, max=2**63 - 1, step=1, force_input=True),
                io.Combo.Input("rating", options=["s", "q", "e", "g"], default="s"),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.String.Output(display_name="tags"),
                io.String.Output(display_name="post id"),
                io.String.Output(display_name="post url"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, seed, rating) -> str:
        username, api_key = get_credentials()
        return json.dumps(
            {
                "seed": seed,
                "rating": rating,
                "username": username,
                "api_key": bool(api_key),
            },
            sort_keys=True,
        )

    @classmethod
    def execute(cls, seed, rating) -> io.NodeOutput:
        del seed
        username, api_key = get_credentials()
        post = fetch_random_post(rating, username, api_key)
        image_url = pick_image_url(post)
        tensor = download_image_tensor(
            image_url,
            username=username,
            api_key=api_key,
        )
        return io.NodeOutput(
            tensor,
            post_tags(post),
            str(post.get("id", "")),
            post_page_url(post),
        )


class DanbooruSearchImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SV-DanbooruSearchImage",
            display_name="Danbooru Search Image",
            category="SV Nodes/Input",
            inputs=[
                io.String.Input("tags", default="1girl", multiline=False),
                io.Boolean.Input("randomize", default=True),
                io.Combo.Input(
                    "rating",
                    options=["(any)", "s", "q", "e", "g"],
                    default="(any)",
                ),
                io.Int.Input("seed", default=0, min=0, max=2**63 - 1, step=1, force_input=True),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.String.Output(display_name="tags"),
                io.String.Output(display_name="post id"),
                io.String.Output(display_name="post url"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        tags,
        randomize,
        rating,
        seed,
    ) -> str:
        username, api_key = get_credentials()
        return json.dumps(
            {
                "tags": tags,
                "randomize": randomize,
                "rating": rating,
                "seed": seed,
                "username": username,
                "api_key": bool(api_key),
            },
            sort_keys=True,
        )

    @classmethod
    def execute(
        cls,
        tags,
        randomize,
        rating,
        seed,
    ) -> io.NodeOutput:
        del seed
        if not isinstance(tags, str) or not tags.strip():
            raise ValueError("tags is required")
        username, api_key = get_credentials()
        rating_filter = None if rating == "(any)" else rating
        post = fetch_post_by_tags(
            tags,
            randomize=bool(randomize),
            rating=rating_filter,
            username=username,
            api_key=api_key,
        )
        image_url = pick_image_url(post)
        tensor = download_image_tensor(
            image_url,
            username=username,
            api_key=api_key,
        )
        return io.NodeOutput(
            tensor,
            post_tags(post),
            str(post.get("id", "")),
            post_page_url(post),
        )

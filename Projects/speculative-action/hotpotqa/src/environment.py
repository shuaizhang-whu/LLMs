"""HotPotQA/FEVER 共用的 Wikipedia 交互环境。

该模块只负责把文本 action 解释为 ``search[...]``、``lookup[...]``、
``finish[...]`` 等环境操作。真实 Wikipedia 搜索和 LLM 生成的模拟搜索结果
都写入同一个 :class:`WikiEnv` 实例，因此理解 ``page``、``obs`` 和 lookup
游标这些可变状态，是判断投机分支是否相互隔离的关键。
"""

import time
import threading
import gymnasium as gym
import requests
from bs4 import BeautifulSoup

from . import constants
from .prompts import PromptTemplates
from .llm_client import LLMClient


def clean_str(p):
    """修复 Wikipedia HTML 文本中经过转义后出现的字符编码问题。"""
    return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


class TextSpace(gym.spaces.Space):
    """最小文本空间：本环境的 observation 与 action 都是 Python 字符串。"""

    def contains(self, x) -> bool:
        """判断 ``x`` 是否属于该文本空间。"""
        return isinstance(x, str)


class WikiEnv(gym.Env):
    """执行 ReAct 风格 Wikipedia action 的底层 Gym 环境。

    ``step_type`` 只改变 ``search[...]`` 的实现：``wiki`` 发起真实 HTTP
    请求，``guess``/``simulate`` 调用 Guess LLM 生成伪页面。其他 action
    不区分真实分支与模拟分支。尤其需要注意，本类没有为每个投机分支复制
    状态；所有调用共享下列 ``page``、lookup 游标和 ``obs``。
    """

    def __init__(self, guess_model_name=None, guess_llm=None):
        super().__init__()
        # 当前页面全文。真实 search 和 Guess LLM 都会写入该字段；lookup 直接
        # 从它构造结果列表，所以一次模拟 search 可能影响之后的真实 lookup。
        self.page = None
        # 最近一次环境调用对外返回的文本 observation，同样由所有分支共享。
        self.obs = None
        # lookup[...] 的易变游标状态：关键词、匹配句列表及下一条结果下标。
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = None
        self.steps = 0
        self.answer = None
        self.observation_space = self.action_space = TextSpace()
        # 这两个计数器跨 episode 累积；reset() 不会将它们清零。
        self.search_time = 0
        self.num_searches = 0
        # simulate search 的截断 observation。普通 guess 不更新该字段，且
        # reset() 也不清空它，调用方应只在对应 simulate step 后立即读取。
        self.sim_obs = None
        self.guess_model_name = guess_model_name
        self.guess_llm = guess_llm
        if self.guess_llm is None and guess_model_name is not None:
            self.guess_llm = LLMClient(
                model_name=guess_model_name,
                temperature=constants.guess_temperature,
                max_tokens=constants.max_guess_output_tokens,
                top_p=constants.guess_top_p,
            )
        # 在线模式只把无副作用 Search 的真实 HTTP 结果放入此缓存。缓存命中后
        # 仍通过正常 env.step 提交权威 Actor 动作，wrapper 奖励和日志语义不变。
        self._prefetch_lock = threading.Lock()
        self._prefetched_search_results = {}
        self.last_search_metadata = None

    def _get_obs(self):
        """返回当前共享 observation。"""
        return self.obs

    def _get_info(self):
        """返回底层环境维护的步数与已提交答案。"""
        return {"steps": self.steps, "answer": self.answer}

    def reset(self, seed=None, return_info=False, options=None):
        """重置单个 episode 的交互状态。

        参数保留 Gym 风格接口，但本实现不使用 ``seed`` 与 ``options``。
        搜索耗时统计和 ``sim_obs`` 是运行级状态，不在这里重置。
        """
        self.obs = ("Interact with Wikipedia using search[], lookup[], and "
                    "finish[].\n")
        self.page = None
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = None
        self.steps = 0
        self.answer = None
        with self._prefetch_lock:
            self._prefetched_search_results.clear()
        self.last_search_metadata = None
        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def construct_lookup_list(self, keyword):
        """从当前 ``page`` 中筛出包含 ``keyword`` 的句子。"""
        if self.page is None:
            return []
        paragraphs = self.page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        sentences = []
        for p in paragraphs:
            sentences += p.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        parts = [p for p in sentences if keyword.lower() in p.lower()]
        return parts

    @staticmethod
    def get_page_obs(page):
        """把页面压缩为前五个句子，作为送回 Agent 的 observation。"""
        paragraphs = page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        sentences = []
        for p in paragraphs:
            sentences += p.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        return ' '.join(sentences[:5])

    def guess_step(self, entity, simulate=False, max_retries=3):
        """让 Guess LLM 为实体生成模拟页面，并写回共享环境状态。

        ``simulate=True`` 只额外保存 ``sim_obs``；无论该标志为何值，
        ``self.page`` 和 ``self.obs`` 都会被模拟页面覆盖。这里并未保存或恢复
        真实 Wikipedia 页面，也没有为 Top-K 分支建立独立缓存。
        """
        if self.guess_llm is None:
            raise RuntimeError("This WikiEnv has no observation speculator configured")
        prompt_wrap = PromptTemplates.GUESS_STEP_PROMPT.format(entity)
        llm_response = None
        for _ in range(max_retries):
            llm_response = self.guess_llm.call(prompt_wrap)
            if llm_response:
                break
        if not llm_response:
            llm_response = ""
        if simulate:
            self.sim_obs = self.get_page_obs(llm_response)
        # 该赋值是共享状态风险的核心：模拟结果会成为后续 lookup 的数据源。
        self.page = llm_response
        self.obs = self.get_page_obs(self.page)

    @staticmethod
    def _search_cache_key(action):
        """只规范化动作名称，保留参数大小写与空白以执行严格字符串命中。"""
        action = action.strip()
        bracket = action.find("[")
        if bracket < 0:
            return action
        return action[:bracket].lower() + action[bracket:]

    @staticmethod
    def fetch_search_result(entity):
        """执行无状态 Wikipedia Search，并返回可稍后应用的真实结果。

        此函数不读取或修改 ``WikiEnv`` 实例，可安全放入线程池。返回对象保留
        HTTP 耗时、请求数、页面正文和 observation，供在线预取与普通 Search
        共用同一解析实现。
        """
        entity_ = entity.replace(" ", "+")
        search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
        started = time.perf_counter()
        requests_headers = {"User-Agent": "React/1.0"}
        response_text = requests.get(search_url, headers=requests_headers).text
        elapsed = time.perf_counter() - started
        soup = BeautifulSoup(response_text, features="html.parser")
        result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
        if result_divs:
            result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
            return {
                "requested_entity": entity,
                "resolved_entity": entity,
                "observation": f"Could not find {entity}. Similar: {result_titles[:5]}.",
                "page": None,
                "result_titles": result_titles,
                "fetch_seconds": elapsed,
                "num_requests": 1,
            }

        page_parts = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
        if any("may refer to:" in part for part in page_parts):
            nested = WikiEnv.fetch_search_result("[" + entity + "]")
            nested["requested_entity"] = entity
            nested["fetch_seconds"] += elapsed
            nested["num_requests"] += 1
            return nested

        page = ""
        for part in page_parts:
            if len(part.split(" ")) > 2:
                page += clean_str(part)
                if not part.endswith("\n"):
                    page += "\n"
        return {
            "requested_entity": entity,
            "resolved_entity": entity,
            "observation": WikiEnv.get_page_obs(page),
            "page": page,
            "result_titles": [],
            "fetch_seconds": elapsed,
            "num_requests": 1,
        }

    def install_prefetched_search(self, action, result, prefetch_metadata=None):
        """把真实 Search 结果安装到严格动作键下，等待权威 env.step 消费。"""
        key = self._search_cache_key(action)
        with self._prefetch_lock:
            self._prefetched_search_results[key] = {
                "result": result,
                "prefetch_metadata": dict(prefetch_metadata or {}),
            }

    def discard_prefetched_searches(self):
        """清空尚未由权威 Actor 命中的预取结果。"""
        with self._prefetch_lock:
            self._prefetched_search_results.clear()

    def apply_search_result(self, result, prefetched=False, prefetch_metadata=None):
        """将纯 Search 结果应用到权威环境状态。"""
        self.search_time += result["fetch_seconds"]
        self.num_searches += result["num_requests"]
        if result["result_titles"]:
            self.result_titles = result["result_titles"]
            self.obs = result["observation"]
        else:
            self.page = result["page"]
            self.obs = result["observation"]
            self.lookup_keyword = self.lookup_list = self.lookup_cnt = None
        self.last_search_metadata = {
            "prefetched": bool(prefetched),
            "fetch_seconds": result["fetch_seconds"],
            "num_requests": result["num_requests"],
            **dict(prefetch_metadata or {}),
        }

    def search_step(self, entity, action=None):
        """同步请求或消费预取的 Wikipedia Search，并更新真实 observation。

        ``search_time`` 只覆盖 HTTP 请求本身，不包含随后 HTML 解析的耗时。
        当搜索页给出相似标题列表时，本分支只更新 ``obs``，不会清空先前
        ``page``/lookup 状态；成功取得正文时才会重置 lookup 游标。
        """
        action = action or f"search[{entity}]"
        key = self._search_cache_key(action)
        with self._prefetch_lock:
            cached = self._prefetched_search_results.pop(key, None)
        if cached is not None:
            self.apply_search_result(
                cached["result"],
                prefetched=True,
                prefetch_metadata=cached["prefetch_metadata"],
            )
            return
        result = self.fetch_search_result(entity)
        self.apply_search_result(result, prefetched=False)

    def step(self, action, step_type="wiki"):
        """解释并执行一条文本 action。

        Args:
            action: ReAct action 字符串。
            step_type: ``wiki`` 表示真实搜索；``guess`` 表示 LLM 模拟搜索；
                ``simulate`` 也执行 LLM 模拟，并把截断结果记入 ``sim_obs``。

        ``step_type`` 只在 action 为 ``search[...]`` 时被读取。``lookup``、
        ``finish``、``think`` 均直接操作同一套共享状态，这意味着调用者把
        正常轨迹和模拟轨迹交错送入同一个环境时，二者并非状态隔离。
        """
        reward = 0
        done = False
        action = action.strip()
        if self.answer is not None:
            done = True
            return self.obs, reward, done, self._get_info()

        if action.startswith("search[") and action.endswith("]"):
            entity = action[len("search["):-1]
            if step_type == "wiki":
                self.search_step(entity, action=action)
            elif step_type == "guess":
                self.guess_step(entity)
            elif step_type == "simulate":
                self.guess_step(entity, simulate=True)
            else:
                raise ValueError("Run is not valid. Step type needs to be wiki or guess")
        elif action.startswith("lookup[") and action.endswith("]"):
            keyword = action[len("lookup["):-1]
            # 切换关键词会基于“此刻的共享 page”重建列表；相同关键词则沿用
            # 现有列表和计数器，不会感知 page 是否已被另一分支覆盖。
            if self.lookup_keyword != keyword:
                self.lookup_keyword = keyword
                self.lookup_list = self.construct_lookup_list(keyword)
                self.lookup_cnt = 0
            if self.lookup_cnt >= len(self.lookup_list):
                self.obs = "No more results.\n"
            else:
                self.obs = f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) " + self.lookup_list[self.lookup_cnt]
                self.lookup_cnt += 1
        elif action.startswith("finish[") and action.endswith("]"):
            answer = action[len("finish["):-1]
            self.answer = answer
            done = True
            self.obs = f"Episode finished, reward = {reward}\n"
        elif action.startswith("think[") and action.endswith("]"):
            self.obs = "Nice thought."
        else:
            self.obs = "Invalid action: {}".format(action)

        self.steps += 1
        return self.obs, reward, done, self._get_info()

    def get_time_info(self):
        """返回本环境累计的真实 Wikipedia HTTP 调用耗时统计。"""
        speed = self.search_time / self.num_searches if self.num_searches else 0
        return {
            "call_speed": speed,
            "call_time": self.search_time,
            "num_calls": self.num_searches,
        }

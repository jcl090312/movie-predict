# app.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------
st.set_page_config(page_title="영화 흥행 예측기", page_icon="🎬", layout="wide")

DAILY_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_daily.csv"
MOVIES_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"


# ------------------------------------------------------------
# 데이터 불러오기
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    daily = pd.read_csv(DAILY_URL, encoding="utf-8")
    movies = pd.read_csv(MOVIES_URL, encoding="utf-8")
    return daily, movies


def find_column(df, candidates, required=True):
    """여러 가능성이 있는 열 이름 중 실제 데이터에 존재하는 열을 찾는다."""
    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise ValueError(
            f"필요한 열을 찾을 수 없습니다.\n"
            f"찾으려 한 열 이름: {candidates}\n"
            f"현재 열 이름: {list(df.columns)}"
        )
    return None


def to_number(series):
    """쉼표 등이 포함된 문자열도 숫자로 바꾼다."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).replace("nan", np.nan),
        errors="coerce",
    )


def make_movie_table(daily, movies):
    """
    일별 박스오피스 표를 영화 단위로 요약한 뒤,
    영화 정보 표와 movieCd를 기준으로 결합한다.
    """
    daily = daily.copy()
    movies = movies.copy()

    # 영화 정보 표의 핵심 열 확인
    movie_cd_col = find_column(movies, ["movieCd", "영화코드"])
    total_audi_col = find_column(movies, ["total_audi", "총 관객", "총관객수"])
    open_dt_col = find_column(movies, ["openDt", "개봉일"], required=False)
    genre_col = find_column(movies, ["genre", "장르"], required=False)
    nation_col = find_column(movies, ["nation", "국가"], required=False)

    # 일별 표의 열 찾기: 안내된 한글 열 이름과 영문 가능성을 모두 지원
    daily_movie_cd_col = find_column(daily, ["movieCd", "영화코드"])
    daily_date_col = find_column(daily, ["날짜", "date", "targetDt"])
    daily_rank_col = find_column(daily, ["순위", "rank", "rankOldAndNew"], required=False)
    daily_scrn_col = find_column(daily, ["스크린수", "scrnCnt"], required=False)
    daily_show_col = find_column(daily, ["상영횟수", "showCnt"], required=False)

    # 자료형 정리
    movies = movies.rename(columns={movie_cd_col: "movieCd"})
    movies["movieCd"] = movies["movieCd"].astype(str)

    movies["total_audi"] = to_number(movies[total_audi_col])

    if "first_scrn" in movies.columns:
        movies["first_scrn"] = to_number(movies["first_scrn"])
    if "first_show" in movies.columns:
        movies["first_show"] = to_number(movies["first_show"])
    if "first_week_audi" in movies.columns:
        movies["first_week_audi"] = to_number(movies["first_week_audi"])
    if "days_in_top10" in movies.columns:
        movies["days_in_top10"] = to_number(movies["days_in_top10"])
    if "peak" in movies.columns:
        movies["peak"] = to_number(movies["peak"])

    # 개봉일에서 연도·월 추출
    if open_dt_col is not None:
        open_date = pd.to_datetime(
            movies[open_dt_col].astype(str), format="%Y%m%d", errors="coerce"
        )
        movies["open_year"] = open_date.dt.year
        movies["open_month"] = open_date.dt.month
    else:
        movies["open_year"] = np.nan
        movies["open_month"] = np.nan

    # 범주형 열 이름을 통일
    if genre_col is not None:
        movies["genre"] = movies[genre_col].fillna("정보없음").astype(str)
    else:
        movies["genre"] = "정보없음"

    if nation_col is not None:
        movies["nation"] = movies[nation_col].fillna("정보없음").astype(str)
    else:
        movies["nation"] = "정보없음"

    # 일별 표 전처리
    daily = daily.rename(columns={daily_movie_cd_col: "movieCd"})
    daily["movieCd"] = daily["movieCd"].astype(str)
    daily["_date"] = pd.to_datetime(
        daily[daily_date_col].astype(str), format="%Y%m%d", errors="coerce"
    )

    # 일별 데이터에서 영화별 보조 설명 변수 생성
    aggregation_dict = {
        "daily_observed_days": ("movieCd", "size"),
    }

    if daily_rank_col is not None:
        daily["_rank"] = to_number(daily[daily_rank_col])
        aggregation_dict["daily_best_rank"] = ("_rank", "min")

    if daily_scrn_col is not None:
        daily["_scrn"] = to_number(daily[daily_scrn_col])
        aggregation_dict["daily_max_scrn"] = ("_scrn", "max")

    if daily_show_col is not None:
        daily["_show"] = to_number(daily[daily_show_col])
        aggregation_dict["daily_max_show"] = ("_show", "max")

    daily_summary = (
        daily.groupby("movieCd", as_index=False)
        .agg(**aggregation_dict)
    )

    # 영화 정보 표의 모든 영화를 기준으로 LEFT JOIN
    movie_table = movies.merge(daily_summary, on="movieCd", how="left")

    # 중복 영화코드가 있을 경우 첫 행만 사용
    movie_table = movie_table.drop_duplicates(subset=["movieCd"]).copy()

    period_start = daily["_date"].min()
    period_end = daily["_date"].max()

    return movie_table, period_start, period_end


# ------------------------------------------------------------
# 화면
# ------------------------------------------------------------
st.title("🎬 영화 흥행 예측기")
st.write(
    "영화 정보와 KOBIS 일별 박스오피스 자료를 결합하여 "
    "**영화의 총 관객 수(total_audi)** 를 예측하는 다중 회귀 모델입니다."
)

try:
    with st.spinner("KOBIS 데이터를 불러오고 있습니다..."):
        daily_df, movies_df = load_data()
        movie_table, period_start, period_end = make_movie_table(daily_df, movies_df)
except Exception as e:
    st.error("데이터를 불러오거나 처리하는 중 문제가 발생했습니다.")
    st.exception(e)
    st.stop()

# 사용할 수 있는 변수 목록
feature_info = {
    "first_scrn": "첫 관측일 스크린 수",
    "first_show": "첫 관측일 상영 횟수",
    "peak": "성수기 개봉 여부(1월·7월·12월)",
    "first_week_audi": "첫 주 관객 수",
    "days_in_top10": "10위권에 있었던 일수",
    "open_year": "개봉 연도",
    "open_month": "개봉 월",
    "genre": "장르",
    "nation": "국가",
    "daily_observed_days": "일별 자료에서 관측된 일수",
    "daily_best_rank": "일별 자료 중 가장 높은 순위(숫자가 작을수록 높음)",
    "daily_max_scrn": "일별 자료 중 최대 스크린 수",
    "daily_max_show": "일별 자료 중 최대 상영 횟수",
}

available_features = [
    feature for feature in feature_info
    if feature in movie_table.columns
]

default_features = {
    "first_scrn",
    "first_show",
    "peak",
    "first_week_audi",
    "days_in_top10",
    "open_month",
    "genre",
    "nation",
}

st.sidebar.header("학습 변수 선택")
st.sidebar.write("체크한 변수를 사용하여 총 관객 수를 예측합니다.")

selected_features = []
for feature in available_features:
    checked = st.sidebar.checkbox(
        feature_info[feature],
        value=(feature in default_features),
        key=f"feature_{feature}",
    )
    if checked:
        selected_features.append(feature)

if len(selected_features) == 0:
    st.warning("왼쪽 사이드바에서 학습에 사용할 변수를 적어도 하나 선택하세요.")
    st.stop()

# 목표값이 있는 영화는 모두 사용
model_df = movie_table[movie_table["total_audi"].notna()].copy()
model_df = model_df.sort_values("movieCd").reset_index(drop=True)

# 영화코드 순서에서 10편마다 앞 3편: 시험용
model_df["dataset"] = np.where(model_df.index % 10 < 3, "시험", "학습")

train_df = model_df[model_df["dataset"] == "학습"].copy()
test_df = model_df[model_df["dataset"] == "시험"].copy()

if len(train_df) == 0 or len(test_df) == 0:
    st.error("학습용 또는 시험용 영화가 부족합니다.")
    st.stop()

# ------------------------------------------------------------
# 데이터 기준 정보
# ------------------------------------------------------------
period_text = "확인 불가"
if pd.notna(period_start) and pd.notna(period_end):
    period_text = f"{period_start.strftime('%Y-%m-%d')} ~ {period_end.strftime('%Y-%m-%d')}"

info1, info2, info3, info4 = st.columns(4)
info1.metric("학습에 쓴 영화", f"{len(train_df):,}편")
info2.metric("점수를 잰 시험용 영화", f"{len(test_df):,}편")
info3.metric("영화 정보 표의 영화", f"{len(movie_table):,}편")
info4.metric("일별 박스오피스 기준 기간", period_text)

st.caption(
    "분할 규칙: movieCd 기준 오름차순 정렬 후, 매 10편 중 앞의 3편은 시험용이고 "
    "나머지 7편은 학습용입니다."
)

# ------------------------------------------------------------
# 회귀 모델 만들기
# ------------------------------------------------------------
categorical_features = [
    col for col in selected_features
    if col in ["genre", "nation"]
]
numeric_features = [
    col for col in selected_features
    if col not in categorical_features
]

transformers = []

if numeric_features:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    transformers.append(("num", numeric_transformer, numeric_features))

if categorical_features:
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformers.append(("cat", categorical_transformer, categorical_features))

preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regression", LinearRegression()),
    ]
)

X_train = train_df[selected_features]
y_train = train_df["total_audi"]
X_test = test_df[selected_features]
y_test = test_df["total_audi"]

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# ------------------------------------------------------------
# 점수와 오차 계산
# ------------------------------------------------------------
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)
r2 = r2_score(y_test, predictions)

positive_target_mask = y_test > 0
if positive_target_mask.sum() > 0:
    mape = mean_absolute_percentage_error(
        y_test[positive_target_mask],
        predictions[positive_target_mask],
    ) * 100
else:
    mape = np.nan

st.subheader("시험용 영화 예측 점수")

score1, score2, score3, score4 = st.columns(4)
score1.metric("R² 결정계수", f"{r2:.3f}")
score2.metric("MAE 평균 절대 오차", f"{mae:,.0f}명")
score3.metric("RMSE 제곱근 평균제곱오차", f"{rmse:,.0f}명")
score4.metric("MAPE 평균 절대 백분율 오차", f"{mape:.1f}%")

st.write("**선택한 학습 변수:** " + ", ".join(feature_info[x] for x in selected_features))

# ------------------------------------------------------------
# 시험용 영화 산점도
# ------------------------------------------------------------
plot_df = test_df[["movieCd"]].copy()
plot_df["actual"] = y_test.values
plot_df["predicted"] = predictions
plot_df["error"] = plot_df["predicted"] - plot_df["actual"]
plot_df["absolute_error"] = plot_df["error"].abs()
plot_df["error_rate_percent"] = np.where(
    plot_df["actual"] > 0,
    plot_df["error"] / plot_df["actual"] * 100,
    np.nan,
)

# 로그 축에서 예측값이 1,000명 미만이면 1,000 위치에 표시
plot_df["predicted_for_plot"] = np.maximum(plot_df["predicted"], 1000)
below_1000_count = int((plot_df["predicted"] < 1000).sum())

# 로그 축을 위해 실제 관객 수가 0 이하인 영화는 그래프에서 제외
scatter_df = plot_df[plot_df["actual"] > 0].copy()
scatter_df["actual_for_plot"] = np.maximum(scatter_df["actual"], 1)

st.subheader("실제 총 관객 수와 예측 총 관객 수")

st.info(
    f"예측 관객 수가 1,000명보다 작은 영화는 **{below_1000_count:,}편**입니다. "
    "이 영화들은 로그 그래프에서 y=1,000 위치(그래프 바닥)에 붙여 표시했습니다."
)

if len(scatter_df) > 0:
    min_line = min(
        scatter_df["actual_for_plot"].min(),
        scatter_df["predicted_for_plot"].min(),
        1000,
    )
    max_line = max(
        scatter_df["actual_for_plot"].max(),
        scatter_df["predicted_for_plot"].max(),
        1000,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=scatter_df["actual_for_plot"],
            y=scatter_df["predicted_for_plot"],
            mode="markers",
            name="시험용 영화",
            marker=dict(
                size=10,
                color=scatter_df["absolute_error"],
                colorscale="Bluered",
                showscale=True,
                colorbar=dict(title="절대 오차(명)"),
                line=dict(color="white", width=0.5),
            ),
            customdata=np.stack(
                [
                    scatter_df["movieCd"],
                    scatter_df["actual"],
                    scatter_df["predicted"],
                    scatter_df["error"],
                    scatter_df["error_rate_percent"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>영화코드: %{customdata[0]}</b><br>"
                "실제 총 관객: %{customdata[1]:,.0f}명<br>"
                "예측 총 관객: %{customdata[2]:,.0f}명<br>"
                "오차(예측-실제): %{customdata[3]:,.0f}명<br>"
                "오차율: %{customdata[4]:.1f}%<extra></extra>"
            ),
        )
    )

    # 실제값 = 예측값 기준선
    fig.add_trace(
        go.Scatter(
            x=[min_line, max_line],
            y=[min_line, max_line],
            mode="lines",
            name="실제값 = 예측값",
            line=dict(color="black", dash="dash", width=2),
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        height=650,
        xaxis_title="실제 총 관객 수 (로그 축)",
        yaxis_title="예측 총 관객 수 (로그 축)",
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=50, r=30, t=40, b=50),
    )
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("실제 총 관객 수가 0보다 큰 시험용 영화가 없어 산점도를 그릴 수 없습니다.")

# ------------------------------------------------------------
# 영화별 예측 결과 표
# ------------------------------------------------------------
st.subheader("시험용 영화별 예측 결과")

result_df = plot_df[
    [
        "movieCd",
        "actual",
        "predicted",
        "error",
        "absolute_error",
        "error_rate_percent",
    ]
].copy()

result_df.columns = [
    "영화코드",
    "실제 총 관객 수",
    "예측 총 관객 수",
    "오차(예측-실제)",
    "절대 오차",
    "오차율(%)",
]

result_df = result_df.sort_values("영화코드").reset_index(drop=True)

st.dataframe(
    result_df.style.format(
        {
            "실제 총 관객 수": "{:,.0f}",
            "예측 총 관객 수": "{:,.0f}",
            "오차(예측-실제)": "{:,.0f}",
            "절대 오차": "{:,.0f}",
            "오차율(%)": "{:.1f}",
        }
    ),
    use_container_width=True,
    height=450,
)

# ------------------------------------------------------------
# 전체 영화 데이터 확인 표
# ------------------------------------------------------------
with st.expander("결합된 영화별 데이터 확인"):
    display_columns = (
        ["movieCd", "total_audi", "dataset"]
        + selected_features
    )
    display_columns = [x for x in display_columns if x in model_df.columns]

    st.dataframe(
        model_df[display_columns].sort_values("movieCd"),
        use_container_width=True,
        height=400,
    )

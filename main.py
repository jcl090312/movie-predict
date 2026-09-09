# main.py
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


# ============================================================
# 1. 앱 설정
# ============================================================
st.set_page_config(
    page_title="영화 흥행 예측기",
    page_icon="🎬",
    layout="wide",
)

DAILY_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_daily.csv"
MOVIES_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"

TARGET_COLUMN = "total_audi"

# 비교할 기본 모델의 변수 3개
BASIC_FEATURES = [
    "first_scrn",
    "first_show",
    "peak",
]

# 기본 변수 3개에 첫 주 관객 수를 더한 모델
BASIC_PLUS_WEEK_FEATURES = [
    "first_scrn",
    "first_show",
    "peak",
    "first_week_audi",
]


# ============================================================
# 2. 데이터 처리 함수
# ============================================================
@st.cache_data(show_spinner=False)
def load_data():
    """GitHub 주소에서 CSV 파일 두 개를 불러온다."""
    daily = pd.read_csv(DAILY_URL, encoding="utf-8")
    movies = pd.read_csv(MOVIES_URL, encoding="utf-8")
    return daily, movies


def find_column(df, candidates, required=True):
    """
    후보 열 이름 목록에서 실제 데이터프레임에 존재하는 열 이름을 찾는다.
    """
    for column in candidates:
        if column in df.columns:
            return column

    if required:
        raise ValueError(
            f"필요한 열을 찾을 수 없습니다.\n"
            f"찾으려 한 열 후보: {candidates}\n"
            f"현재 데이터의 열: {list(df.columns)}"
        )

    return None


def to_number(series):
    """문자열, 쉼표가 포함된 수 등을 숫자로 변환한다."""
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .replace("nan", np.nan),
        errors="coerce",
    )


def make_movie_table(daily, movies):
    """
    일별 박스오피스 데이터를 영화별로 요약하고,
    영화 정보 표와 movieCd를 기준으로 합친다.

    영화 정보 표(movies)의 모든 영화가 유지되도록 left join을 사용한다.
    """
    daily = daily.copy()
    movies = movies.copy()

    # --------------------------------------------------------
    # 영화 정보 표 열 확인
    # --------------------------------------------------------
    movie_cd_col = find_column(movies, ["movieCd", "영화코드"])
    total_audi_col = find_column(movies, ["total_audi", "총 관객", "총관객수"])

    open_dt_col = find_column(movies, ["openDt", "개봉일"], required=False)
    genre_col = find_column(movies, ["genre", "장르"], required=False)
    nation_col = find_column(movies, ["nation", "국가"], required=False)

    # movieCd 열 이름 통일
    movies = movies.rename(columns={movie_cd_col: "movieCd"})
    movies["movieCd"] = movies["movieCd"].astype(str)

    # 목표값: 총 관객 수
    movies["total_audi"] = to_number(movies[total_audi_col])

    # 수치형 영화 정보 열 숫자화
    numeric_movie_columns = [
        "first_scrn",
        "first_show",
        "peak",
        "first_week_audi",
        "days_in_top10",
    ]

    for column in numeric_movie_columns:
        if column in movies.columns:
            movies[column] = to_number(movies[column])

    # 개봉일에서 개봉 연도와 개봉 월 생성
    if open_dt_col is not None:
        open_date = pd.to_datetime(
            movies[open_dt_col].astype(str),
            format="%Y%m%d",
            errors="coerce",
        )
        movies["open_year"] = open_date.dt.year
        movies["open_month"] = open_date.dt.month
    else:
        movies["open_year"] = np.nan
        movies["open_month"] = np.nan

    # 장르와 국가 처리
    if genre_col is not None:
        movies["genre"] = movies[genre_col].fillna("정보없음").astype(str)
    else:
        movies["genre"] = "정보없음"

    if nation_col is not None:
        movies["nation"] = movies[nation_col].fillna("정보없음").astype(str)
    else:
        movies["nation"] = "정보없음"

    # --------------------------------------------------------
    # 일별 박스오피스 표 열 확인
    # --------------------------------------------------------
    daily_movie_cd_col = find_column(daily, ["movieCd", "영화코드"])
    daily_date_col = find_column(daily, ["날짜", "date", "targetDt"])
    daily_rank_col = find_column(daily, ["순위", "rank"], required=False)
    daily_scrn_col = find_column(daily, ["스크린수", "scrnCnt"], required=False)
    daily_show_col = find_column(daily, ["상영횟수", "showCnt"], required=False)

    # movieCd 열 이름 통일
    daily = daily.rename(columns={daily_movie_cd_col: "movieCd"})
    daily["movieCd"] = daily["movieCd"].astype(str)

    # 날짜 형식 변환
    daily["_date"] = pd.to_datetime(
        daily[daily_date_col].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )

    # 일별 자료에서 영화별 추가 변수 생성
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

    # 일별 자료를 영화코드별로 집계
    daily_summary = (
        daily.groupby("movieCd", as_index=False)
        .agg(**aggregation_dict)
    )

    # 영화 정보 표의 모든 영화를 유지하며 결합
    movie_table = movies.merge(
        daily_summary,
        on="movieCd",
        how="left",
    )

    # 혹시 movieCd 중복이 있으면 첫 번째 행만 남긴다.
    movie_table = movie_table.drop_duplicates(subset=["movieCd"]).copy()

    # 일별 박스오피스 자료의 기간
    period_start = daily["_date"].min()
    period_end = daily["_date"].max()

    return movie_table, period_start, period_end


# ============================================================
# 3. 회귀 모델 함수
# ============================================================
def build_model(selected_features):
    """
    선택한 변수에 맞게 결측치 처리와 원-핫 인코딩을 포함한
    선형 회귀 모델 파이프라인을 만든다.
    """
    categorical_features = [
        feature
        for feature in selected_features
        if feature in ["genre", "nation"]
    ]

    numeric_features = [
        feature
        for feature in selected_features
        if feature not in categorical_features
    ]

    transformers = []

    # 수치형 열: 중앙값으로 결측치 채우기
    if numeric_features:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        transformers.append(
            ("numeric", numeric_transformer, numeric_features)
        )

    # 범주형 열: 최빈값으로 결측치 채우고 원-핫 인코딩
    if categorical_features:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )
        transformers.append(
            ("categorical", categorical_transformer, categorical_features)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regression", LinearRegression()),
        ]
    )

    return model


def evaluate_model(train_df, test_df, selected_features):
    """
    학습용 영화로 모델을 학습하고,
    시험용 영화의 예측값과 평가 점수를 반환한다.
    """
    model = build_model(selected_features)

    x_train = train_df[selected_features]
    y_train = train_df[TARGET_COLUMN]

    x_test = test_df[selected_features]
    y_test = test_df[TARGET_COLUMN]

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    mae = mean_absolute_error(y_test, predictions)

    # 최신 scikit-learn 호환:
    # squared=False 대신 평균제곱오차의 제곱근을 직접 계산한다.
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    r2 = r2_score(y_test, predictions)

    # 실제값이 0보다 큰 경우만 MAPE 계산에 사용
    positive_mask = y_test > 0

    if positive_mask.sum() > 0:
        mape = (
            mean_absolute_percentage_error(
                y_test[positive_mask],
                predictions[positive_mask],
            )
            * 100
        )
    else:
        mape = np.nan

    return {
        "model": model,
        "predictions": predictions,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
    }


# ============================================================
# 4. 화면 시작
# ============================================================
st.title("🎬 영화 흥행 예측기")

st.write(
    "KOBIS 영화 정보와 일별 박스오피스 자료를 결합하여 "
    "**영화별 총 관객 수(total_audi)** 를 예측하는 다중 회귀 분석 앱입니다."
)

st.warning(
    """
    **해석 주의: 이 결과는 실제 개봉 전 흥행 예측 성능이 아닙니다.**

    이 데이터에는 첫 주 관객 수, 10위권 유지 일수, 일별 최대 스크린 수,
    일별 최대 상영 횟수처럼 영화가 개봉한 뒤에 확인할 수 있는
    **사후 집계값**이 포함되어 있습니다.

    따라서 이 앱은 개봉 전 예측기라기보다,
    개봉 후 수집된 정보와 총 관객 수의 관계를 살펴보는
    **사후 분석용 회귀 모델**로 해석해야 합니다.
    """
)

try:
    with st.spinner("KOBIS 데이터를 불러오고 처리하고 있습니다..."):
        daily_df, movies_df = load_data()
        movie_table, period_start, period_end = make_movie_table(
            daily_df,
            movies_df,
        )

except Exception as error:
    st.error("데이터를 불러오거나 처리하는 중 오류가 발생했습니다.")
    st.exception(error)
    st.stop()


# ============================================================
# 5. 학습용·시험용 데이터 분리
# ============================================================
# 목표값(total_audi)이 있는 영화만 회귀 모델 평가에 사용한다.
model_df = movie_table[
    movie_table[TARGET_COLUMN].notna()
].copy()

# movieCd 순서 정렬
model_df = model_df.sort_values("movieCd").reset_index(drop=True)

# 매 10편 중 앞 3편: 시험용 / 나머지 7편: 학습용
model_df["dataset"] = np.where(
    model_df.index % 10 < 3,
    "시험",
    "학습",
)

train_df = model_df[model_df["dataset"] == "학습"].copy()
test_df = model_df[model_df["dataset"] == "시험"].copy()

if len(train_df) == 0 or len(test_df) == 0:
    st.error("학습용 또는 시험용 영화가 부족하여 모델을 만들 수 없습니다.")
    st.stop()


# ============================================================
# 6. 데이터 기준 정보
# ============================================================
if pd.notna(period_start) and pd.notna(period_end):
    period_text = (
        f"{period_start.strftime('%Y-%m-%d')} ~ "
        f"{period_end.strftime('%Y-%m-%d')}"
    )
else:
    period_text = "확인 불가"

info1, info2, info3, info4 = st.columns(4)

info1.metric("학습에 쓴 영화", f"{len(train_df):,}편")
info2.metric("점수를 잰 시험용 영화", f"{len(test_df):,}편")
info3.metric("영화 정보 표의 영화", f"{len(movie_table):,}편")
info4.metric("일별 자료 기준 기간", period_text)

st.caption(
    "분할 방법: movieCd를 오름차순으로 정렬한 뒤, "
    "매 10편마다 앞의 3편은 시험용으로 사용하고 나머지 7편은 학습용으로 사용했습니다."
)


# ============================================================
# 7. 기본 모델과 첫 주 관객 추가 모델 비교
# ============================================================
st.subheader("기본 변수 모델과 첫 주 관객 추가 모델 비교")

required_comparison_features = list(
    set(BASIC_FEATURES + BASIC_PLUS_WEEK_FEATURES)
)

missing_features = [
    feature
    for feature in required_comparison_features
    if feature not in model_df.columns
]

if missing_features:
    st.error(
        "비교 모델을 만들기 위한 변수가 없습니다: "
        + ", ".join(missing_features)
    )
else:
    basic_result = evaluate_model(
        train_df,
        test_df,
        BASIC_FEATURES,
    )

    basic_plus_week_result = evaluate_model(
        train_df,
        test_df,
        BASIC_PLUS_WEEK_FEATURES,
    )

    comparison_df = pd.DataFrame(
        {
            "모델": [
                "기본 변수 3개",
                "기본 변수 3개 + 첫 주 관객 수",
            ],
            "사용 변수": [
                "first_scrn, first_show, peak",
                "first_scrn, first_show, peak, first_week_audi",
            ],
            "R² 결정계수": [
                basic_result["r2"],
                basic_plus_week_result["r2"],
            ],
            "MAE 평균 절대 오차(명)": [
                basic_result["mae"],
                basic_plus_week_result["mae"],
            ],
            "RMSE 제곱근 평균제곱오차(명)": [
                basic_result["rmse"],
                basic_plus_week_result["rmse"],
            ],
            "MAPE 평균 절대 백분율 오차(%)": [
                basic_result["mape"],
                basic_plus_week_result["mape"],
            ],
        }
    )

    st.dataframe(
        comparison_df.style.format(
            {
                "R² 결정계수": "{:.3f}",
                "MAE 평균 절대 오차(명)": "{:,.0f}",
                "RMSE 제곱근 평균제곱오차(명)": "{:,.0f}",
                "MAPE 평균 절대 백분율 오차(%)": "{:.1f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "R²는 클수록 좋으며, MAE·RMSE·MAPE는 작을수록 좋습니다. "
        "첫 주 관객 수는 개봉 후에 알 수 있는 정보이므로, "
        "첫 주 관객 수를 포함한 모델을 개봉 전 예측 모델로 해석해서는 안 됩니다."
    )


# ============================================================
# 8. 사용자 변수 선택
# ============================================================
feature_info = {
    "first_scrn": "첫 관측일 스크린 수",
    "first_show": "첫 관측일 상영 횟수",
    "peak": "성수기 개봉 여부",
    "first_week_audi": "첫 주 관객 수 (사후 정보)",
    "days_in_top10": "10위권에 있었던 일수 (사후 정보)",
    "open_year": "개봉 연도",
    "open_month": "개봉 월",
    "genre": "장르",
    "nation": "국가",
    "daily_observed_days": "일별 자료에서 관측된 일수 (사후 정보)",
    "daily_best_rank": "일별 자료의 최고 순위 (사후 정보)",
    "daily_max_scrn": "일별 자료의 최대 스크린 수 (사후 정보)",
    "daily_max_show": "일별 자료의 최대 상영 횟수 (사후 정보)",
}

available_features = [
    feature
    for feature in feature_info
    if feature in model_df.columns
]

st.sidebar.header("학습 변수 선택")
st.sidebar.write("체크한 변수를 사용해 총 관객 수를 예측합니다.")

selected_features = []

for feature in available_features:
    # 기본 선택값: 기본 3개 + 첫 주 관객 수
    is_default = feature in BASIC_PLUS_WEEK_FEATURES

    is_checked = st.sidebar.checkbox(
        feature_info[feature],
        value=is_default,
        key=f"feature_{feature}",
    )

    if is_checked:
        selected_features.append(feature)

if len(selected_features) == 0:
    st.warning("왼쪽 사이드바에서 학습에 사용할 변수를 하나 이상 선택하세요.")
    st.stop()


# ============================================================
# 9. 선택 변수 모델 학습 및 점수
# ============================================================
selected_result = evaluate_model(
    train_df,
    test_df,
    selected_features,
)

predictions = selected_result["predictions"]

st.subheader("선택한 변수로 만든 회귀 모델")

st.write(
    "**선택한 학습 변수:** "
    + ", ".join(feature_info[feature] for feature in selected_features)
)

score1, score2, score3, score4 = st.columns(4)

score1.metric(
    "R² 결정계수",
    f"{selected_result['r2']:.3f}",
)

score2.metric(
    "MAE 평균 절대 오차",
    f"{selected_result['mae']:,.0f}명",
)

score3.metric(
    "RMSE 제곱근 평균제곱오차",
    f"{selected_result['rmse']:,.0f}명",
)

score4.metric(
    "MAPE 평균 절대 백분율 오차",
    f"{selected_result['mape']:.1f}%",
)


# ============================================================
# 10. 시험용 영화의 예측 결과 표 만들기
# ============================================================
plot_df = test_df[["movieCd"]].copy()

plot_df["actual"] = test_df[TARGET_COLUMN].values
plot_df["predicted"] = predictions
plot_df["error"] = plot_df["predicted"] - plot_df["actual"]
plot_df["absolute_error"] = plot_df["error"].abs()

plot_df["error_rate_percent"] = np.where(
    plot_df["actual"] > 0,
    plot_df["error"] / plot_df["actual"] * 100,
    np.nan,
)

# 로그 축에서는 0 또는 음수 값을 표시할 수 없다.
# 예측값이 1,000명보다 작으면 그래프의 바닥인 1,000에 붙여 표시한다.
plot_df["predicted_for_plot"] = np.maximum(
    plot_df["predicted"],
    1000,
)

below_1000_count = int(
    (plot_df["predicted"] < 1000).sum()
)

# 실제 관객 수가 0 이하인 경우에는 로그 x축에 그릴 수 없으므로 제외한다.
scatter_df = plot_df[
    plot_df["actual"] > 0
].copy()

scatter_df["actual_for_plot"] = np.maximum(
    scatter_df["actual"],
    1,
)


# ============================================================
# 11. Plotly 산점도
# ============================================================
st.subheader("실제 총 관객 수와 예측 총 관객 수")

st.info(
    f"예측 총 관객 수가 1,000명보다 작은 시험용 영화는 "
    f"**{below_1000_count:,}편**입니다. "
    "이 영화들은 로그 축 그래프에서 y=1,000 위치(그래프 바닥)에 붙여 표시했습니다."
)

if len(scatter_df) > 0:
    line_min = min(
        scatter_df["actual_for_plot"].min(),
        scatter_df["predicted_for_plot"].min(),
        1000,
    )

    line_max = max(
        scatter_df["actual_for_plot"].max(),
        scatter_df["predicted_for_plot"].max(),
        1000,
    )

    fig = go.Figure()

    # 시험용 영화 산점도
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
                "실제 총 관객 수: %{customdata[1]:,.0f}명<br>"
                "예측 총 관객 수: %{customdata[2]:,.0f}명<br>"
                "오차(예측-실제): %{customdata[3]:,.0f}명<br>"
                "오차율: %{customdata[4]:.1f}%"
                "<extra></extra>"
            ),
        )
    )

    # 실제값 = 예측값을 뜻하는 y = x 기준선
    fig.add_trace(
        go.Scatter(
            x=[line_min, line_max],
            y=[line_min, line_max],
            mode="lines",
            name="실제값 = 예측값",
            line=dict(
                color="black",
                dash="dash",
                width=2,
            ),
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
    st.warning(
        "실제 총 관객 수가 0보다 큰 시험용 영화가 없어 산점도를 그릴 수 없습니다."
    )


# ============================================================
# 12. 시험용 영화별 예측 결과 표
# ============================================================
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


# ============================================================
# 13. 결합된 영화별 자료 확인
# ============================================================
with st.expander("결합된 영화별 데이터 확인"):
    display_columns = [
        "movieCd",
        "total_audi",
        "dataset",
    ] + selected_features

    # 중복 열 제거 및 실제 존재하는 열만 선택
    display_columns = list(dict.fromkeys(display_columns))
    display_columns = [
        column
        for column in display_columns
        if column in model_df.columns
    ]

    st.dataframe(
        model_df[display_columns]
        .sort_values("movieCd")
        .reset_index(drop=True),
        use_container_width=True,
        height=400,
    )

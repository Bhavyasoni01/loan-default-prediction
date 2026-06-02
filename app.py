#fastapi 

from fastapi import FastAPI
from pydantic import BaseModel
import joblib 
import pandas as pd
from datetime import datetime,timezone

app = FastAPI()

with open('saved_model.pkl', 'rb') as f:
    model = joblib.load(f)
class Loan(BaseModel):
    loan_amnt: float
    funded_amnt: float
    funded_amnt_inv: float
    term: float
    int_rate: float
    installment: float
    grade: str
    sub_grade: str
    emp_length: float
    home_ownership: str
    annual_inc: float
    verification_status: str
    issue_d: str
    purpose: str
    addr_state: str
    dti: float
    delinq_2yrs: float
    inq_last_6mths: float
    open_acc: float
    pub_rec: float
    revol_bal: float
    revol_util: float
    total_acc: float
    initial_list_status: str
    collections_12_mths_ex_med: float
    acc_now_delinq: float
    tot_coll_amt: float
    tot_cur_bal: float
    total_rev_hi_lim: float
    acc_open_past_24mths: float
    avg_cur_bal: float
    bc_open_to_buy: float
    bc_util: float
    chargeoff_within_12_mths: float
    delinq_amnt: float
    mo_sin_old_il_acct: float
    mo_sin_old_rev_tl_op: float
    mo_sin_rcnt_rev_tl_op: float
    mo_sin_rcnt_tl: float
    mort_acc: float
    mths_since_recent_bc: float
    mths_since_recent_inq: float
    num_accts_ever_120_pd: float
    num_actv_bc_tl: float
    num_actv_rev_tl: float
    num_bc_sats: float
    num_bc_tl: float
    num_il_tl: float
    num_op_rev_tl: float
    num_rev_accts: float
    num_rev_tl_bal_gt_0: float
    num_sats: float
    num_tl_120dpd_2m: float
    num_tl_30dpd: float
    num_tl_90g_dpd_24m: float
    num_tl_op_past_12m: float
    pct_tl_nvr_dlq: float
    percent_bc_gt_75: float
    pub_rec_bankruptcies: float
    tax_liens: float
    tot_hi_cred_lim: float
    total_bal_ex_mort: float
    total_bc_limit: float
    total_il_high_credit_limit: float
    debt_to_installment: float
    fico_avg: float
    funded_ratio: float
    revol_util_bucket: str
    payment_to_income: float
    delinq_ratio: float
    credit_intensity: float
    open_acc_ratio: float


@app.get("/")
def home():
    return {
        "message": "Loan Defaulter Prediction Model "
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "loan-default-api",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/predict")
def predict(loan : Loan):
    data = pd.DataFrame([loan.model_dump()])

    probabilty  = float((model.predict_proba(data)[0][1]))
    prediction = "default" if probabilty >= 0.45 else "no_default"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    return {
        'prediction':prediction,
        'default_probability_percent' : round(probabilty * 100,2),
        'timestamp' : timestamp
    }

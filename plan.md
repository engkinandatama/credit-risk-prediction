# RINGKASAN EKSEKUTIF & RENCANA AKSI (Berdasarkan EDA)


### TARGET VARIABLE ###
   1. Definisi 'Bad Loan' (risk=1) adalah status: 'Charged Off', 'Default', 'Late (31-120 days)', dll.
   2. Definisi 'Good Loan' (risk=0) adalah 'Fully Paid'.
   3. Status pinjaman yang masih berjalan ('Current', 'In Grace Period') akan difilter dan tidak digunakan dalam pemodelan.
------------------------------------------------------------
### COLUMNS TO DROP ###
   1. Kolom dengan >40% nilai hilang: inq_fi, open_rv_24m, max_bal_bc, all_util, inq_last_12m, annual_inc_joint, verification_status_joint, dti_joint, total_cu_tl, il_util, mths_since_rcnt_il, total_bal_il, open_il_24m, open_il_12m, open_il_6m, open_acc_6m, open_rv_12m, mths_since_last_record, mths_since_last_major_derog, desc, mths_since_last_delinq, next_pymnt_d.
   2. Kolom yang mengandung kebocoran data (informasi masa depan): funded_amnt, funded_amnt_inv, total_pymnt, total_rec_prncp, total_rec_int, last_pymnt_d, last_pymnt_amnt.
   3. Kolom redundan karena multikolinearitas tinggi (>0.7): total_rec_prncp, collection_recovery_fee, out_prncp_inv, mths_since_last_major_derog, funded_amnt, revol_bal, last_pymnt_amnt, total_rec_int, loan_amnt, total_pymnt, out_prncp, mths_since_last_delinq, funded_amnt_inv, installment, total_rev_hi_lim, total_pymnt_inv, recoveries, member_id, id (contoh: 'loan_amnt' vs 'funded_amnt').
   4. Kolom ID ('id', 'member_id') dan teks bebas ('desc', 'title') akan dihapus.
------------------------------------------------------------
### FEATURE ENGINEERING IDEAS ###
   1. Buat 'credit_history_length' dari 'issue_d' dan 'earliest_cr_line'.
   2. Buat rasio-rasio penting: 'loan_to_income', 'installment_to_income'.
   3. Terapkan transformasi log pada fitur numerik yang miring (skewed) seperti 'annual_inc' dan 'revol_bal'.
------------------------------------------------------------
### PREPROCESSING STRATEGY ###
   1. Imputasi nilai hilang pada fitur numerik menggunakan median (lebih robust terhadap outlier).
   2. Imputasi nilai hilang pada fitur kategorikal menggunakan modus ('missing' atau nilai paling umum).
   3. Terapkan StandardScaler pada semua fitur numerik setelah imputasi.
   4. Terapkan OneHotEncoder pada fitur kategorikal setelah imputasi.
   5. Gunakan ColumnTransformer untuk memastikan semua langkah preprocessing digabungkan dalam satu pipeline yang konsisten.
------------------------------------------------------------

### LOG TEMUAN OTOMATIS ###
   - LOG: Terdeteksi 22 kolom dengan nilai hilang > 40%.
   - LOG: Terdeteksi 19 kolom redundan karena korelasi tinggi.
============================================================

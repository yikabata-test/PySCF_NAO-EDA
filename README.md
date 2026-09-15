# PySCF_NAO-EDA

PySCF 上でエネルギー密度解析 (Energy Density Analysis, EDA) を行い，分子の全エネルギーを
原子ごとのエネルギーに分割するためのコードです。

参考文献:

* H. Nakai, "Energy density analysis with Kohn–Sham orbitals",
  *Chem. Phys. Lett.* **363**, 73–79 (2002). — 従来型 EDA
* T. Baba, M. Takeuchi, H. Nakai, "Natural atomic orbital based energy density
  analysis: Implementation and applications",
  *Chem. Phys. Lett.* **424**, 193–198 (2006). — NAO-EDA, LSO-EDA

現在は **閉殻 Hartree–Fock (RHF)** に対する従来型 EDA，LSO-EDA，NAO-EDA を
実装しています (`pyscf_eda.rhf`, `pyscf_eda.orth`)。

## インストール

```bash
pip install -e .          # 開発用インストール
pip install -e .[test]    # テストも実行する場合
pytest
```

## 使い方

```python
from pyscf import gto, scf
from pyscf_eda import rhf as eda_rhf

mol = gto.M(atom='O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586', basis='cc-pvdz')
mf = scf.RHF(mol).run()

res = eda_rhf.EDA(mf).kernel()   # 従来型 EDA (または eda_rhf.kernel(mf))
print(res.summary())             # 成分ごとの原子エネルギーの表
print(res.e_tot)                 # 原子の全エネルギー (numpy 配列, hartree)

res_nao = eda_rhf.EDA(mf, orbital_basis='nao').kernel()   # NAO-EDA (eda_rhf.nao_eda(mf))
res_lso = eda_rhf.EDA(mf, orbital_basis='lso').kernel()   # LSO-EDA (eda_rhf.lso_eda(mf))
print(res_nao.pop)               # 自然電荷 (NPA) に対応する原子電子数
```

`res` には次の原子ごとの配列 (長さ `mol.natm`, 単位 hartree) が入っています。

| 属性      | 内容                                            |
|-----------|-------------------------------------------------|
| `e_nn`    | 核間反発エネルギー $E_{NN}^A$                    |
| `e_kin`   | 運動エネルギー $T_S^A$                           |
| `e_ne`    | 核–電子引力エネルギー $E_{Ne}^A$ (ECP を含む)    |
| `e_1el`   | 一電子エネルギー $E_{1EL}^A = T_S^A + E_{Ne}^A$   |
| `e_coul`  | クーロンエネルギー $E_{CLB}^A$                   |
| `e_x`     | 交換エネルギー $E_X^A$                           |
| `e_elec`  | 電子エネルギー $E_{ELC}^A$                       |
| `e_tot`   | 全エネルギー $E_{TOT}^A = E_{NN}^A + E_{ELC}^A$  |
| `pop`     | 同じ軌道基底での原子電子数 (MPA / LPA / NPA)      |
| `orth_coeff` | 用いた変換行列 $X$ (AO → 直交化原子軌道)       |

$\sum_A E_{TOT}^A$ は SCF の全エネルギーと (数値精度の範囲で) 厳密に一致します。
`kernel()` の実行時にこの和則を検査し，ずれがあれば警告を出します。

## 分割の定義

RHF の全エネルギー

$$E_{TOT} = E_{NN} + T_S + E_{Ne} + E_{CLB} + E_X$$

を以下のように原子 $A$ の寄与に分割します。$P$ は AO 密度行列，$\mu \in A$ は原子 $A$ に
中心をもつ基底関数を表します。

* **基底関数に基づく分割 (Mulliken 型, 論文の式 (5)–(9))**
  $$T_S^A = \sum_{\mu\in A}(PT)_{\mu\mu},\qquad
    E_{CLB}^A = \tfrac12\sum_{\mu\in A}(PJ)_{\mu\mu},\qquad
    E_X^A = -\tfrac14\sum_{\mu\in A}(PK)_{\mu\mu}$$
* **原子核に基づく分割 (論文の式 (11))**
  $$E_{NN}^A = \tfrac12\sum_{B\neq A}\frac{Z_AZ_B}{R_{AB}}$$
* **核–電子引力 (既定: `ne_partition='half'`)**: 半分を基底関数，半分を原子核で分割します。
  $$E_{Ne}^A = \tfrac12\sum_{\mu\in A}(PV)_{\mu\mu}
             + \tfrac12\sum_{\mu\nu}P_{\mu\nu}(V_A)_{\nu\mu},\qquad
    V=\sum_A V_A,\quad (V_A)_{\nu\mu} = -Z_A\langle\nu|\,|r-R_A|^{-1}|\mu\rangle$$
  原著論文どおり全てを基底関数で分割する場合は `ne_partition='mulliken'`，
  全てを原子核で分割する場合は `ne_partition='nuclear'` を指定します。
  有効内殻ポテンシャル (ECP) も核–電子相互作用として同じ規則で分割されます。

## NAO-EDA / LSO-EDA (軌道基底の選択)

従来型 EDA の基底関数による分割は Mulliken population analysis と同じ性質をもち，
基底関数依存性が大きくなります。Baba–Takeuchi–Nakai (2006) に従い，AO を直交化した
一中心軌道 $\phi_l = \sum_m X_{ml}\chi_m$ ($X^\dagger S X = 1$) の基底で同じ分割を行います。

$$P' = X^{-1} P X^{-1\dagger},\qquad M' = X^\dagger M X,\qquad
  E^A[M] = \sum_{l\in A}(P'M')_{ll}$$

`orbital_basis` 引数で軌道基底を選びます。

| `orbital_basis` | 内容 | 対応する population |
|---|---|---|
| `'ao'` (既定) | 生の AO ($X=1$)，従来型 EDA | Mulliken (MPA) |
| `'nao'` | 自然原子軌道 (NAO)，NAO-EDA。`pyscf.lo.nao` の OWSO 変換を使用 | Natural (NPA) |
| `'lso'` / `'lowdin'` | Löwdin 対称直交化 ($X=S^{-1/2}$)，LSO-EDA | Löwdin (LPA) |
| `'meta_lowdin'` | PySCF の meta-Löwdin 軌道 (参考) | meta-Löwdin |
| `numpy.ndarray` | 任意の変換行列 $X$ | – |

運動エネルギー，クーロン，交換，および核–電子引力の基底関数分割部分がこの基底で
評価されます。核間反発と核–電子引力の原子核分割部分は基底に依存しません。
どの軌道基底でも $\sum_A E_{TOT}^A$ は SCF 全エネルギーと一致します。

CO2 (RHF, R = 1.16 Å) の C 原子のエネルギー比率 $E_C/E_{total}$ は，
NAO-EDA では STO-3G を除き 20.8–20.9 % でほぼ一定であり，LSO-EDA では diffuse 関数で
外れ値が現れるなど，論文 Table 2 と同じ傾向が得られます (`examples/co2_nao_eda.py`)。

## 例

* `examples/h2o_rhf_eda.py` – 論文 Table 1 と同じ H2O 構造 (cc-pVDZ) での RHF-EDA。
  孤立原子からの差 (Table 1 の括弧内に相当) は，孤立原子を通常の UHF で計算して求めます
  (単一原子ではその全エネルギーがそのまま原子エネルギーになるため EDA は不要です)。
* `examples/h2o_bond_breaking.py` – O–H 結合伸長 (2002 年論文 Table 2 に対応) に伴う原子エネルギーの変化
* `examples/co2_nao_eda.py` – CO2 での従来型 / LSO- / NAO-EDA の基底関数依存性 (2006 年論文 Table 1, 2 に対応)

## 制限事項

* 現在は閉殻 RHF のみ対応しています (UHF/ROHF, KS-DFT は未対応で例外を出します)。
  検証などで孤立原子のエネルギーが必要な場合は，通常の UHF 計算を用いてください。
  密度フィッティング (`.density_fit()`) や ECP を用いた RHF には対応しています。
* `mf.get_hcore()` に $T + V_{nuc} (+V_{ECP})$ 以外の一電子項 (外場など) が含まれる場合，
  その寄与は基底関数で分割し `e_other` として報告します。

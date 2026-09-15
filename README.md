# PySCF_NAO-EDA

PySCF 上でエネルギー密度解析 (Energy Density Analysis, EDA) を行い，分子の全エネルギーを
原子ごとのエネルギーに分割するためのコードです。

参考文献:

* H. Nakai, "Energy density analysis with Kohn–Sham orbitals",
  *Chem. Phys. Lett.* **363**, 73–79 (2002). — 従来型 EDA
* T. Baba, M. Takeuchi, H. Nakai, "Natural atomic orbital based energy density
  analysis: Implementation and applications",
  *Chem. Phys. Lett.* **424**, 193–198 (2006). — NAO-EDA, LSO-EDA

* M. Kobayashi, Y. Imamura, H. Nakai, "Alternative linear-scaling methodology for the
  second-order Møller–Plesset perturbation calculation based on the divide-and-conquer
  method", *J. Chem. Phys.* **127**, 074103 (2007). — MP2 相関エネルギーの分割 (Sec. II B)
* M. Kobayashi, H. Nakai, "Extension of linear-scaling divide-and-conquer-based correlation
  method to coupled cluster theory with singles and doubles excitations",
  *J. Chem. Phys.* **129**, 044103 (2008). — CCSD 相関エネルギーの分割 (Sec. II B)
* M. Kobayashi, H. Nakai, "Divide-and-conquer-based linear-scaling approach for traditional and
  renormalized coupled cluster methods with single, double, and noniterative triple excitations",
  *J. Chem. Phys.* **131**, 114108 (2009). — (T) 補正の分割 (Sec. II B)
* J. Seino, H. Nakai, "Informatics-based energy fitting scheme for correlation energy at
  complete basis set limit", *J. Comput. Chem.* **37**, 2304–2315 (2016). — QDD / QTD 等の
  CBS 極限のフィッティングモデル

現在は **閉殻 Hartree–Fock (RHF)**，**閉殻 MP2**，**閉殻 CCSD**，**閉殻 CCSD(T)** に対する
従来型 EDA，LSO-EDA，NAO-EDA と，それらを組み合わせた **完全基底極限 (CBS) の原子エネルギー**
(QDD / QTD / QTT / QTN) を実装しています
(`pyscf_eda.rhf`, `pyscf_eda.mp2`, `pyscf_eda.ccsd`, `pyscf_eda.ccsd_t`, `pyscf_eda.cbs`,
`pyscf_eda.orth`)。

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

res = eda_rhf.EDA(mf).kernel()   # NAO-EDA (既定; または eda_rhf.kernel(mf))
print(res.summary())             # 成分ごとの原子エネルギーの表
print(res.e_tot)                 # 原子の全エネルギー (numpy 配列, hartree)
print(res.pop)                   # 自然電荷 (NPA) に対応する原子電子数

res_ao = eda_rhf.EDA(mf, orbital_basis='ao').kernel()     # 従来型 (Mulliken 型) EDA
res_lso = eda_rhf.EDA(mf, orbital_basis='lso').kernel()   # LSO-EDA (eda_rhf.lso_eda(mf))
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

`orbital_basis` 引数で軌道基底を選びます。**すべての EDA (RHF / MP2 / CCSD / CCSD(T) / CBS) で
既定は NAO-EDA (`'nao'`) です。**

| `orbital_basis` | 内容 | 対応する population |
|---|---|---|
| `'nao'` (既定) | 自然原子軌道 (NAO)，NAO-EDA。`pyscf.lo.nao` の OWSO 変換を使用 | Natural (NPA) |
| `'ao'` | 生の AO ($X=1$)，従来型 (Mulliken 型) EDA | Mulliken (MPA) |
| `'lso'` / `'lowdin'` | Löwdin 対称直交化 ($X=S^{-1/2}$)，LSO-EDA | Löwdin (LPA) |
| `'meta_lowdin'` | PySCF の meta-Löwdin 軌道 (参考) | meta-Löwdin |
| `numpy.ndarray` | 任意の変換行列 $X$ | – |

運動エネルギー，クーロン，交換，および核–電子引力の基底関数分割部分がこの基底で
評価されます。核間反発と核–電子引力の原子核分割部分は基底に依存しません。
どの軌道基底でも $\sum_A E_{TOT}^A$ は SCF 全エネルギーと一致します。

### 実装の検証と基底依存性についての注意

* 実装は論文の式 (6), (7) を PySCF の NAO (`pyscf.lo.orth_ao(mf, 'nao')`) でそのまま評価した
  独立計算と 1e-8 以内で一致し，NAO 基底での電子数は PySCF の NPA と一致します
  (`tests/test_rhf_eda.py`)。原子ごとの部分トレースは原子内のユニタリ回転に不変なので，
  NAO の「natural character の復元」の有無は結果に影響しません。
* CO2 (RHF, R = 1.16 Å, cc-pVDZ) の C 原子のエネルギー比率 $E_C/E_{total}$ は NAO-EDA で
  20.91 %，従来型で 20.10 % であり，論文 Table 2 (B3LYP: 20.92 %, 20.15 %) をよく再現します。
  NAO-EDA は STO-3G を除き 20.8–20.9 % でほぼ一定です (`examples/co2_nao_eda.py`)。
* 論文で従来型 EDA に現れた外れ値 (aug-cc-pVTZ で $E_C/E_{total}$ = 19.72 %, Mulliken 電子数 4.4)
  は，PySCF 既定の球面調和型基底では現れず (20.13 %, 5.52)，GAMESS 既定のデカルト型基底
  (`cart=True`) にすると再現されます (20.03 %, 4.28)。NAO-EDA はデカルト/球面で不変 (20.79 %) です。
  論文が示す NAO-EDA の優位性の大部分は，このような Mulliken 型分割の破綻を NAO が回避する点に
  あります。
* 一方，Mulliken 型分割が破綻しない条件では NAO-EDA が常に基底依存性が小さいわけではありません。
  H2O (RHF, 球面調和基底 11 種, STO-3G を除く) の O 原子エネルギーの標準偏差は従来型 0.03，
  NAO 0.05，LSO 0.21 hartree で，NAO-EDA は Pople 系基底の間ではほぼ一定 (−74.41 ± 0.02) ですが
  cc-pVXZ 系では −74.34〜−74.56 と変動します。また NAO-EDA は陽イオン的な原子に大きく負の
  エネルギーを与えます (H2O の H: −0.84 hartree; 論文の CO2 でも C に 1.45 hartree 移動)。
  これらは手法固有の性質で，用途に応じて `orbital_basis='ao'` も検討してください。

## MP2-EDA

DC-MP2 論文 (Kobayashi–Imamura–Nakai 2007) の Sec. II B は，MP2 相関エネルギーを
「最後の AO→MO 変換を残す」ことで原子に分割します。サブシステムを 1 原子，バッファ領域を
全系にとれば DC の式 (20)–(22) は通常の MP2 に帰着するので，式 (18) をそのまま実装しています。

$$E_{corr} = \sum_{ij}^{occ}\sum_{ab}^{vir}(ia|jb)\,(2t_{ijab}-t_{ijba}),\qquad
  t_{ijab} = \frac{(ia|jb)}{\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b}$$

$$E_{corr}^A = \sum_{ij}^{occ}\sum_{ab}^{vir}\Big[w_{occ}\sum_{\mu\in A}C_{\mu i}(\mu a|jb)
  + w_{vir}\sum_{\mu\in A}C_{\mu a}(i\mu|jb)\Big](2t_{ijab}-t_{ijba}),\qquad w_{occ}+w_{vir}=1$$

$(pq|rs)$ は化学者の記法の二電子積分です。論文の数値検証に従い既定は $w_{occ}=1$
(占有軌道側のみで分割) で，`w_occ` 引数で変更できます。原子の MP2 エネルギーは
$E_{MP2}^A = E_{HF}^A + E_{corr}^A$ で，$\sum_A E_{MP2}^A$ は MP2 全エネルギーと一致します。
frozen core (`mp.MP2(mf, frozen=...)`) と密度フィッティング MP2 に対応しています。

```python
from pyscf import gto, scf, mp
from pyscf_eda import mp2 as eda_mp2

mf = scf.RHF(mol).run()
pt = mp.MP2(mf, frozen=1).run()
res = eda_mp2.EDA(pt).kernel()                      # NAO 基底での分割 (既定)
res = eda_mp2.EDA(pt, orbital_basis='ao').kernel()  # 従来型 (AO) 分割
print(res.summary())
print(res.e_hf, res.e_corr, res.e_tot)              # 原子ごとの HF / 相関 / MP2 エネルギー
```

`res` には RHF-EDA の全成分に加えて `e_hf`，`e_corr_occ` (占有側分割)，`e_corr_vir` (仮想側分割)，
`e_corr` ($=w_{occ}e_{corr}^{occ}+w_{vir}e_{corr}^{vir}$)，`e_tot` ($=E_{MP2}^A$) が入ります。
NAO-EDA では AO 添字 $\mu$ を直交化軌道 $\phi_l$ に置き換え，$C\to X^{-1}C$，
$(\mu a|jb)\to(la|jb)$ として同じ式を評価します。

## CCSD-EDA

DC-CCSD 論文 (Kobayashi–Nakai 2008) の式 (10) は MP2 と同じ形で，有効振幅 $\tilde t$ を
CCSD の T1・T2 振幅から作ります。サブシステムを 1 原子，バッファを全系にとれば
式 (11)–(14) は通常の CCSD に帰着します。

$$E_{corr}^A = \sum_{ij}^{occ}\sum_{ab}^{vir}\Big[w_{occ}\sum_{\mu\in A}C_{\mu i}(\mu a|jb)
  + w_{vir}\sum_{\mu\in A}C_{\mu a}(i\mu|jb)\Big](2\tau_{ijab}-\tau_{ijba}),\qquad
  \tau_{ijab} = t_{ijab} + t_{ia}t_{jb}$$

**注意:** 論文の式 (8) は $\tilde t_{ij,ab} = t_{ia}t_{jb} - t_{ib}t_{ja} + t_{ij,ab}$ と
書かれていますが，この反対称化した形はスピン軌道表記のものであり，閉殻の空間軌道の式 (6) に
そのまま代入すると CCSD 相関エネルギーを再現しません。閉殻 CCSD のエネルギー式
(PySCF の実装と同じ) に対応する $\tau_{ijab} = t_{ijab} + t_{ia}t_{jb}$ を用いることで
$\sum_A E_{corr}^A = E_{corr}$ が厳密に成り立ちます (`tests/test_ccsd_eda.py` で確認)。
一般の CCSD エネルギー式に現れる一重励起項 $2\sum_{ia}f_{ia}t_{ia}$ は正準 HF 軌道ではゼロで，
論文の式 (6) にも含まれないため既定では分割しません (`with_singles=True` で含められます)。
既定は論文の式 (14) と同じく $w_{occ}=1$ です。

```python
from pyscf import cc
from pyscf_eda import ccsd as eda_ccsd

mycc = cc.CCSD(mf, frozen=1).run()
res = eda_ccsd.EDA(mycc).kernel()                      # NAO 基底での分割 (既定)
res = eda_ccsd.EDA(mycc, orbital_basis='ao').kernel()  # 従来型 (AO) 分割
print(res.summary())
print(res.e_hf, res.e_corr, res.e_tot)                 # 原子ごとの HF / 相関 / CCSD エネルギー
```

結果オブジェクトの属性は MP2-EDA と同じです (`e_tot` が $E_{CCSD}^A$)。

## CCSD(T)-EDA

DC-CCSD(T) 論文 (Kobayashi–Nakai 2009) の Sec. II B に従います。(T) 補正は 4 次の三重励起項
$E_T^{[4]}$ と 5 次の一重–三重励起項 $E_{ST}^{[5]}$ からなり (式 (9), (10))，両者は共通因子
$t^{(2)}_{ijk,abc}D_{ijk,abc}$ (以下 $W$) を持ちます。化学者の記法で

$$W_{ijk}^{abc} = P_{ijk}^{abc}\Big[\sum_e t_{ij}^{ae}(be|ck) - \sum_m t_{im}^{ab}(mj|ck)\Big],\qquad
  Z_{ijk}^{abc} = t_{ia}(bj|ck) + t_{jb}(ai|ck) + t_{kc}(ai|bj)$$

$$E_{(T)} = \sum_{ijk}\sum_{abc}\frac{\tfrac43 T_{abc} - 2T_{acb} + \tfrac23 T_{bca}}{D_{ijk,abc}}\,W_{ijk}^{abc},
  \qquad T = W + Z$$

です ($P$ は (ia),(jb),(kc) の 3 対の対称化演算子)。分割は共通因子 $W$ に対して行い，その
3/4 変換積分中の 1 つの MO を原子 $A$ の基底関数 (または NAO) の部分和に置き換えます (式 (21)–(27))。
論文で採用された $U^{0,0}$ (占有軌道 $k$ の分割) を既定とし，仮想軌道 $c$ を分割する $U^{2,2}$ も
`w_occ` で混合できます ($X=1$, $Y=1$ の変種は未実装)。対称化演算子を係数側に移すと
$P\,Y = 2\,r_3(T)/D$ となるので，原子ごとには非対称化の生の項 $R^A$ だけを作ればよく，

$$E_{(T)}^A = 2\sum_{ijk}\sum_{abc}\frac{[r_3(W+Z)]_{ijk}^{abc}}{D_{ijk,abc}}\,R^{A,abc}_{ijk},\qquad
  R^A = \sum_e t_{ij}^{ae}(be|c\,k_A) - \sum_m t_{im}^{ab}(mj|c\,k_A)$$

として評価します。$\sum_A E_{(T)}^A = E_{(T)}$ は PySCF の `ccsd_t()` と 1e-9 以内で一致します。

```python
from pyscf_eda import ccsd_t as eda_ccsd_t

mycc = cc.CCSD(mf, frozen=1).run()
res = eda_ccsd_t.EDA(mycc).kernel()           # E_HF^A + E_corr(CCSD)^A + E_(T)^A
print(res.e_ccsd, res.e_t, res.e_t4, res.e_st5, res.e_tot)
```

計算量・メモリは通常の (T) と同程度 ($O(o^3v^4)$，仮想軌道ブロックごとに $O(o^3v^2)$ と
原子ごとの部分変換積分 $O(N_{atom}\,o\,v^3)$) なので，小〜中規模分子向けです。
論文の renormalized CCSD(T) (R-CCSD(T)) は未実装です。

## CBS 極限の原子エネルギー (QDD / QTD)

Seino–Nakai (2016) の 3 スキーム線形フィッティング (3SLF) は，階層的基底 cc-pVXZ / aug-cc-pVXZ
での相関エネルギーの線形結合で CCSD(T)/CBS の相関エネルギーを推定します (式 (21))。

$$E_{corr}^{CBS} = \sum_X c^L_X E_{MP2}[X] + \sum_X c^M_X E_{CCSD}[X] + \sum_X c^H_X E_{CCSD(T)}[X]$$

| モデル | 使用するエネルギー | 実行する計算 |
|---|---|---|
| QDD | MP2/D,T,Q; CCSD/D; CCSD(T)/D | CCSD(T)/DZ，MP2/TZ，MP2/QZ |
| QTD | MP2/D,T,Q; CCSD/D,T; CCSD(T)/D | CCSD(T)/DZ，CCSD/TZ，MP2/QZ |
| QTT | MP2/D,T,Q; CCSD/D,T; CCSD(T)/D,T | CCSD(T)/DZ，CCSD(T)/TZ，MP2/QZ |
| QTN | MP2/D,T,Q; CCSD/D,T | CCSD/DZ，CCSD/TZ，MP2/QZ |

係数は論文 Table 11 (cc-pVXZ, aug-cc-pVXZ) の値です。各基底では 1 本の計算の連鎖
(HF → MP2 → CCSD → (T)) から下位レベルのエネルギーも同時に得るので，MP2/DZ や CCSD/DZ の
ための追加計算はありません (MP2/TZ は TZ 基底での計算が必要です)。
モデルもすべての EDA も相関エネルギーについて線形なので，原子ごとの相関エネルギーにも同じ係数を
適用でき，$\sum_A E_{corr}^{CBS,A} = E_{corr}^{CBS}$ が成り立ちます。論文と同じく frozen core
(化学的コア軌道) の相関エネルギーを既定とします。

```python
from pyscf import gto
from pyscf_eda import cbs as eda_cbs

mol = gto.M(atom='O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586')   # 基底は自動で置き換え
res = eda_cbs.QTD(mol).kernel()             # または eda_cbs.CompositeEDA(mol, scheme='QDD')
print(res.summary())
print(res.e_corr)      # CBS 極限の原子相関エネルギー
print(res.e_tot)       # E_HF^A (参照) + E_corr^{CBS,A}
print(res.corr)        # {(手法, 基底): 各レベルの原子相関エネルギー}
print(res.eda_results) # 各レベルの EDA 結果オブジェクト
```

主なオプション: `basis_family='cc-pv' | 'aug-cc-pv'`，`frozen='auto' | int`，
`orbital_basis='nao' | 'ao' | 'lso'`，`ne_partition`，`w_occ`，`hf_cbs`。

### Hartree–Fock エネルギーの CBS 見積り

HF 部分はフィッティングモデルの対象外なので，途中で得られる DZ, TZ, QZ の HF エネルギーから
`hf_cbs` で指定した方法で CBS 極限を見積もります。

| `hf_cbs` | 式 | 性質 |
|---|---|---|
| `'halkier'` (既定) | 2 点 (TZ, QZ) $E(X)=E_{CBS}+A\,e^{-\alpha X}$，$\alpha=1.63$ 固定 (`hf_alpha`) | エネルギーについて線形 |
| `'karton-martin'` | 2 点 (TZ, QZ) $E(X)=E_{CBS}+A\,(X+1)e^{-9\sqrt{X}}$ | 線形 |
| `'feller'` | 3 点 (DZ, TZ, QZ) $E(X)=E_{CBS}+A\,e^{-\alpha X}$，$\alpha$ もフィット: $E_{CBS}=E_Q-(E_Q-E_T)^2/(E_Q-2E_T+E_D)$ | **非線形** |
| `'largest'` | 外挿なし (QZ の値) | – |

線形な 2 点式は原子ごとに適用しても和が分子の外挿値と一致します。Feller の 3 点式は非線形なので，
原子ごとに適用すると和が分子の値と一致せず，原子の HF エネルギーが $X$ に対して単調・幾何級数的に
収束していないとき (例: H2O の H 原子は TZ→QZ で上昇) には分母が小さくなり不安定になります。
検証のため，結果には全方式の見積り (原子ごと・分子全体)，原子和と分子値の差，
Feller のフィット指数 $\alpha_A=\ln[(E_T-E_D)/(E_Q-E_T)]$ (単調収束でなければ nan) が
`res.hf_estimates` に格納され，`summary()` にも表示されます。HF のみを調べるには
`eda_cbs.HFCBS(mol).kernel()` (または `eda_cbs.hf_cbs(mol)`) を使います。

```python
hfres = eda_cbs.HFCBS(mol, basis_family='cc-pv').kernel()   # RHF/DZ,TZ,QZ + EDA のみ
print(hfres.summary())          # 全方式の原子 HF/CBS，原子和 − 分子値，alpha_A
print(hfres.estimates['feller']['atoms'], hfres.estimates['feller_alpha'])
```

## 例

* `examples/h2o_rhf_eda.py` – 論文 Table 1 と同じ H2O 構造 (cc-pVDZ) での RHF-EDA。
  孤立原子からの差 (Table 1 の括弧内に相当) は，孤立原子を通常の UHF で計算して求めます
  (単一原子ではその全エネルギーがそのまま原子エネルギーになるため EDA は不要です)。
* `examples/h2o_bond_breaking.py` – O–H 結合伸長 (2002 年論文 Table 2 に対応) に伴う原子エネルギーの変化
* `examples/co2_nao_eda.py` – CO2 での従来型 / LSO- / NAO-EDA の基底関数依存性 (2006 年論文 Table 1, 2 に対応)
* `examples/h2o_mp2_eda.py` – H2O の MP2-EDA (占有側 / 仮想側分割の比較，NAO 基底，UMP2 孤立原子との差)
* `examples/h2o_ccsd_eda.py` – H2O の CCSD-EDA (MP2-EDA との比較，NAO 基底，UCCSD 孤立原子との差)
* `examples/h2o_ccsd_t_eda.py` – H2O の CCSD(T)-EDA ($U^{0,0}$ / $U^{2,2}$，$E_T^{[4]}$ / $E_{ST}^{[5]}$，UCCSD(T) 孤立原子との差)
* `examples/h2o_cbs_eda.py` – H2O の QDD / QTD による CBS 極限の原子エネルギー
* `examples/h2o_hf_cbs.py` – H2O の原子 HF エネルギーの CBS 見積り (線形 / 非線形外挿の安定性の検証)

## 制限事項

* 現在は閉殻 RHF，閉殻 MP2，閉殻 CCSD，閉殻 CCSD(T) のみ対応しています (UHF/ROHF, UMP2,
  UCCSD, KS-DFT は未対応で例外を出します)。検証などで孤立原子のエネルギーが必要な場合は，
  通常の UHF / UMP2 / UCCSD(T) 計算を用いてください。
* MP2-/CCSD-EDA は原子ごとに半変換積分 $(la|jb)$, $(il|jb)$ を作るため，メモリは
  $O(n_A n_{vir} n_{occ} n_{vir})$，計算量は原子数 × 積分変換のコストになります。
  CCSD-EDA は CCSD の相関エネルギーのみを分割し，(T) 補正は `pyscf_eda.ccsd_t` で扱います。
  密度フィッティング (`.density_fit()`) や ECP を用いた RHF には対応しています。
* `mf.get_hcore()` に $T + V_{nuc} (+V_{ECP})$ 以外の一電子項 (外場など) が含まれる場合，
  その寄与は基底関数で分割し `e_other` として報告します。

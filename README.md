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

現在は **閉殻 Hartree–Fock (RHF)**，**閉殻 MP2**，**閉殻 CCSD** に対する従来型 EDA，LSO-EDA，
NAO-EDA を実装しています (`pyscf_eda.rhf`, `pyscf_eda.mp2`, `pyscf_eda.ccsd`, `pyscf_eda.orth`)。

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
res = eda_mp2.EDA(pt).kernel()                      # 従来型 (AO) 分割
res = eda_mp2.EDA(pt, orbital_basis='nao').kernel() # NAO 基底での分割
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
res = eda_ccsd.EDA(mycc).kernel()                      # 従来型 (AO) 分割
res = eda_ccsd.EDA(mycc, orbital_basis='nao').kernel() # NAO 基底での分割
print(res.summary())
print(res.e_hf, res.e_corr, res.e_tot)                 # 原子ごとの HF / 相関 / CCSD エネルギー
```

結果オブジェクトの属性は MP2-EDA と同じです (`e_tot` が $E_{CCSD}^A$)。

## 例

* `examples/h2o_rhf_eda.py` – 論文 Table 1 と同じ H2O 構造 (cc-pVDZ) での RHF-EDA。
  孤立原子からの差 (Table 1 の括弧内に相当) は，孤立原子を通常の UHF で計算して求めます
  (単一原子ではその全エネルギーがそのまま原子エネルギーになるため EDA は不要です)。
* `examples/h2o_bond_breaking.py` – O–H 結合伸長 (2002 年論文 Table 2 に対応) に伴う原子エネルギーの変化
* `examples/co2_nao_eda.py` – CO2 での従来型 / LSO- / NAO-EDA の基底関数依存性 (2006 年論文 Table 1, 2 に対応)
* `examples/h2o_mp2_eda.py` – H2O の MP2-EDA (占有側 / 仮想側分割の比較，NAO 基底，UMP2 孤立原子との差)
* `examples/h2o_ccsd_eda.py` – H2O の CCSD-EDA (MP2-EDA との比較，NAO 基底，UCCSD 孤立原子との差)

## 制限事項

* 現在は閉殻 RHF，閉殻 MP2，閉殻 CCSD のみ対応しています (UHF/ROHF, UMP2, UCCSD, KS-DFT は
  未対応で例外を出します)。検証などで孤立原子のエネルギーが必要な場合は，
  通常の UHF / UMP2 / UCCSD 計算を用いてください。
* MP2-/CCSD-EDA は原子ごとに半変換積分 $(la|jb)$, $(il|jb)$ を作るため，メモリは
  $O(n_A n_{vir} n_{occ} n_{vir})$，計算量は原子数 × 積分変換のコストになります。
  CCSD-EDA は CCSD の相関エネルギーのみを分割します ((T) 補正は含みません)。
  密度フィッティング (`.density_fit()`) や ECP を用いた RHF には対応しています。
* `mf.get_hcore()` に $T + V_{nuc} (+V_{ECP})$ 以外の一電子項 (外場など) が含まれる場合，
  その寄与は基底関数で分割し `e_other` として報告します。

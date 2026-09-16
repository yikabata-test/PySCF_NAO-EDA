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

現在は **閉殻 Hartree–Fock (RHF)**，**閉殻 Kohn–Sham DFT (RKS)**，**閉殻 MP2**，**閉殻 CCSD**，
**閉殻 CCSD(T)** に対する従来型 EDA，LSO-EDA，NAO-EDA と，それらを組み合わせた
**完全基底極限 (CBS) の原子エネルギー** (QDD / QTD / QTT / QTN) を実装しています
(`pyscf_eda.rhf`, `pyscf_eda.mp2`, `pyscf_eda.ccsd`, `pyscf_eda.ccsd_t`, `pyscf_eda.cbs`,
`pyscf_eda.orth`)。

## インストール

```bash
pip install -e .          # 開発用インストール (C コンパイラがあれば (T) 分割カーネルもビルド)
pip install -e .[test]    # テストも実行する場合
pytest
```

依存パッケージは numpy, scipy, pyscf (>= 2.0), threadpoolctl です。CCSD(T)-EDA の (T) 分割は
`pyscf_eda/lib/ccsd_t_eda.c` の C (OpenMP) カーネルを使います。`pip install` 時に gcc などの
C コンパイラが見つかればビルドされ，見つからなければ純 numpy 実装 (同じ結果，低速) に
自動的にフォールバックします。カーネルが使えているかは次で確認できます。

```python
from pyscf_eda import lib
print(lib.load() is not None, lib.num_threads())   # True, OpenMP スレッド数
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

`summary()` (RHF/RKS, MP2, CCSD, CCSD(T), CBS の各結果) と勾配の表
(`pyscf_eda.grad.format_gradient`) は，原子・分子・CBS 極限のいずれの
エネルギーも勾配も小数点以下 10 桁で出力します (機械学習の教師データ用)。
`examples/` のスクリプトも同じ精度で出力し，numpy 配列は
`numpy.set_printoptions(precision=10, floatmode='fixed')` で表示しています。

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

## Kohn–Sham DFT (RKS) の EDA

`pyscf.dft.RKS` オブジェクトを渡すと DFT の EDA になります。交換相関エネルギーは論文 (2002) の
式 (3) のとおり，グリッド点ごとに Becke 型分割関数を含む重み $w_g p_A(r_g)$ で原子に分割します
(PySCF のグリッドは各点の所属原子 `grids.atm_idx` を保持しています)。ハイブリッド汎関数の
厳密交換は係数 (レンジ分離ハイブリッドを含む) を掛けて基底関数で分割します。
結果には `e_xc` (グリッド分割した汎関数部分) が加わり，`e_x` はハイブリッド係数を掛けた厳密交換です。
VV10 などの非局所相関と分散補正は未対応です。

```python
from pyscf import dft
mf = dft.RKS(mol, xc='B3LYP')
mf.grids = eda_rhf.becke_grids(mf)     # 原著 (HONDO99/GAMESS) と同じ Becke 分割 (Bragg 半径)
mf.run()
res = eda_rhf.EDA(mf).kernel()
print(res.e_xc, res.e_x, res.e_tot)
# 汎関数の成分 (Slater, B88, VWN, LYP など) の原子分割
slater = eda_rhf.xc_energy_by_atom(mf, 'LDA_X')
```

**グリッド分割関数について:** PySCF 既定のグリッドは Treutler–Ahlrichs のサイズ補正を用いますが，
原著の EDA コードは Becke 原法＋Bragg 半径 (H = 0.35 Å) のサイズ補正です。全エネルギーは変わりませんが
原子ごとの $E_{XC}$ が最大 0.1 hartree 程度変わるので，論文と比較する場合は `becke_grids` を使ってください。
さらに GAMESS のソース (`dftgrd.src`, `mod_dft_partfunc.src`) を確認したところ，GAMESS の Becke 分割は
平滑化多項式 $f(x)=\tfrac32 x-\tfrac12 x^3$ を **4 回**反復しており (`BECKE4`)，Becke 原法・HONDO99 の
3 回と異なります。この違いは原子ごとの $E_{XC}$ に最大 0.05 hartree 程度影響します。
`becke_grids(mf, code='hondo')` (3 回, 2002 年論文) と `becke_grids(mf, code='gamess')` (4 回, 2006・2009 年論文)
で切り替えられます (`iterations` で任意回数)。GAMESS 本体の Bragg–Slater 半径表 (H を Bohr 半径 0.529 Å
に変更したもの, `radii='gamess'`) も選べますが，これを使うと文献値から大きく外れるため，中井研の EDA
コードは Becke 原法の半径表 (H = 0.35 Å) を用いていたと判断できます。

### Mulliken-EDA と Grid-EDA (Kikuchi–Imamura–Nakai 2009)

Kikuchi, Imamura, Nakai, *Int. J. Quantum Chem.* **109**, 2464 (2009) は DFT の分割法として
3 種を比較しています。`EDA` の引数で選べます。

| 手法 | T, V_ne, J, K | E_XC | 指定 |
|---|---|---|---|
| 従来型 EDA (N02) | 基底関数 (Mulliken 型) | グリッド (Becke 分割関数) | `xc_partition='grid'` (既定) |
| Mulliken-EDA | 基底関数 | 基底関数: 各グリッド点の密度を Mulliken 型に分割 $\rho_A(r_g)=\sum_{\mu\in A}\sum_\nu P_{\mu\nu}\chi_\mu\chi_\nu$ (式 (30), (31)) | `xc_partition='mulliken'` / `mulliken_eda(mf)` |
| Grid-EDA (fuzzy atom) | すべて Becke 分割関数 (式 (23)–(26); J, K は擬スペクトル法) | グリッド | `all_grid=True` / `grid_eda(mf)` |

Mulliken-EDA の $E_{XC}$ は `orbital_basis` で選んだ軌道基底 (NAO など) で分割されるので，
論文の結論で提案されている「全項目を NAO 基底で分割する」形も `orbital_basis='nao'` で得られます。
Grid-EDA の J, K は各グリッド点の静電ポテンシャル積分 (`int1e_grids`) を用い，級数打ち切りは
グリッド精度 (level 5 で全エネルギーの和則が 1e-6 程度) に依存します。核–電子引力は式 (24) のとおり
半分をグリッド点，半分を原子核で分割します。

### 原著論文との一致 (DFT)

* **H2O, B3LYP/cc-pVDZ (Nakai 2002, Table 1):** デカルト型 d 関数，VWN-RPA 版 B3LYP
  (PySCF の `'B3LYP'`)，Becke–Bragg 分割，核–電子引力の半分ずつの分割 (`ne_partition='half'`) で，
  H・O・分子の全成分 (E_NN, T_S, E_Ne, E_CLB, E_X^HF, Slater, B88, VWN, LYP, E_TOT) が論文の
  全桁 (1e-4 hartree 以内) で一致します (`examples/h2o_b3lyp_table1.py`, `tests/test_rks_eda.py`)。
  核–電子引力を全て基底関数で分割する `'mulliken'` では E_Ne が 0.07 hartree ずれるので，
  原著コードの分割は `'half'` です。
* **G2-1 分子, B3LYP(VWN5)/6-31G(d,p) (Kikuchi–Imamura–Nakai 2009, Table II–III):**
  閉殻分子 (C2H2, C2H4, C2H6, SiH2(1A1), SiH4, HF) について Mulliken-EDA / Grid-EDA / 従来型 EDA
  の原子電子数と原子エネルギーを `examples/g2_mulliken_grid_eda.py` で比較しています。
  GAMESS と同じ 4 回反復の Becke 分割 (`code='gamess'`) を使うと，C2H2, C2H4, C2H6, HF の
  Mulliken-EDA・Grid-EDA・従来型 EDA が論文の全桁 (1–3 mhartree 以内) で一致します
  (SiH2, SiH4 は論文に構造が記載されておらず ~5 mhartree の差が残ります)。
* **CO2, B3LYP (Baba–Takeuchi–Nakai 2006, Table 1–2):** GAMESS 既定の VWN5 版 B3LYP
  (`'B3LYP5'`) とデカルト型関数で全エネルギー (−188.51822 vs −188.51825) と Mulliken 電子数 (5.670)
  が一致します。4 回反復の Becke 分割 (`code='gamess'`) で従来型 EDA の C 原子エネルギーは
  −37.985 vs 論文 −37.98529 と一致します (3 回反復では −0.05 hartree の一律のずれが全基底で残ります)。
  NAO-EDA は NAO の構成法に敏感で，`pyscf.lo.nao` の簡略版では C 原子エネルギーが論文より 0.4 hartree
  低くなりますが，Reed–Weinstock–Weinhold の手順 (NMB を一括 OWSO，NRB を Schmidt 直交化後 OWSO;
  本パッケージの `'nao'`) では論文と数十 mhartree 以内で一致します (NPA 4.96 vs 4.98)。
  基底系列全体の比較は `examples/co2_b3lyp_2006.py` を参照してください。
  残る差は Rydberg 集合 (NRB) の直交化の重み付けなど論文に記載のない NAO 構成の細部に由来し，
  `orbital_basis='nao:pre'` (既定; pre-NAO 占有数を重み), `'nao:post'` (Schmidt 直交化後の対角占有数),
  `'nao:lowdin'` (等重み) で比較できます。CO2 (B3LYP5, GAMESS 分割) の C 原子エネルギーの論文との差は
  例えば cc-pVDZ で +0.054 / −0.019 / −0.072，6-31G で +0.114 / −0.005 / −0.122，
  aug-cc-pVTZ で +0.277 / +0.171 / −0.048 hartree で，全基底を同時に再現する重み付けはありません。

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
| `'nao'` (既定) | 自然原子軌道 (NAO)，NAO-EDA。Reed–Weinstock–Weinhold (1985) の手順 (NMB を一括 OWSO，NRB を Schmidt 直交化後 OWSO) | Natural (NPA) |
| `'nao_pyscf'` | `pyscf.lo.nao` の簡略版 NAO (参考。CO2 の C 原子エネルギーが `'nao'` と 0.4 hartree 異なる) | PySCF NPA |
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
  これらは手法固有の性質です。

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

相関エネルギーの原子分割に必要な半変換積分 $(la|jb)$, $(il|jb)$ は，安価な (占有, 仮想) 対を
先に変換する順序で 1 回 (メモリに応じて占有軌道のブロックごと) にまとめて生成します。
計算量は MP2 自体の $(ia|jb)$ 変換と同じ $O(N^4 o)$ で，原子ごとに変換をやり直していた
以前の実装 ($O(N^5)$) に比べ，メタノール/cc-pVQZ (230 基底関数) で 101 秒 → 10 秒 (1 スレッド) に
短縮されました。使用メモリは `mol.max_memory` で制限されます。
AO 積分がメモリに入らない (outcore) 場合は，ブロックごとに AO 積分を再生成する代わりに
一時 HDF5 ファイルへ 1 回だけ変換してブロックを読み出します ((T) 分割の $(be|cl)$ 等も同様)。
ブタン/cc-pVTZ (260 基底，outcore) の MP2-EDA は 75 秒 → 20 秒 (MP2 本体 20 秒) になりました。
なお，$(la|jb)=\sum_p (C^{-1}X)_{pl}(pa|jb)$ と MO の完全性で書き換えて $(vv|ov)$ 積分を
s4 対称で変換する案は，第 1 半変換のコストが仮想軌道数に比例するため現行より遅く
(同じ系で 29 秒)，採用していません。

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

実装では，原子ごとに $R^A$ を組み立てる代わりに，原子に依存しない縮約
$P_{beck}=\sum_{ija} Y_{ijk}^{abc} t_{ij}^{ae}$, $Q_{mjck}=\sum_{iab} Y_{ijk}^{abc} t_{im}^{ab}$
($Y = 2\,r_3(W+Z)/D$) を先に行い，原子 (軌道 $l$) への帰属は 3/4 変換積分 $(be|cl)$, $(mj|cl)$
(U$^{2,2}$ では $(be|lk)$, $(mj|lk)$) との最後の 1 添字の縮約だけで済ませます。
計算量は $P$ の $O(o^3v^4)$ ((T) 1 回分) と $O(o\,v^3 N)$ で原子数に依存せず，メモリは
$O(o^3v^2)$ (仮想ブロックごと) と $O(o\,v^3)$ ($P$ と $(be|ck)$) で，$(be|cl)$ は `max_memory` に
収まるブロックで生成します。仮想軌道 $a$ ごとのブロック $W_{ijk}^{abc}$ は $b$ についても
`max_memory` に応じて分割するので，大きな分子でもメモリは $O(o^3 v\,n_b)$ に抑えられます。
$W$ の 12 項は転置済みの振幅・積分との行列積として組み立て，末尾の添字 ($c$ または $b,c$) が
連続になる形で累積します。`with_t4=False` (CBS ドライバの既定) では $E_T[4]$ と $E_{ST}[5]$ への
分解を省略します。

$P$, $Q$ の計算には C (OpenMP) で書いたカーネル `pyscf_eda/lib/ccsd_t_eda.c` を使います
(`backend='auto'` 既定。`backend='numpy'` で純 numpy 実装)。PySCF の (T) と同じく仮想軌道の
ブロック $a \ge b \ge c$ ごとに $W$, $Z$, $Y$ を組み立て，その場で $P$, $Q$ へ逆縮約します。
行列積は SciPy が公開する BLAS の `dgemm` を関数ポインタで呼び (なければ C のループ)，
ブロックはスレッドに動的に分配して，共有の $P$, $Q$ は行ごとのロックで更新します。
メモリはスレッドあたり $O(o^3 n_c)$ で，原子数にも仮想軌道数にも依存しません。
カーネルは `pip install` 時にコンパイルされ (C コンパイラがなければ numpy 実装に自動的に
フォールバック)，ソース配布のまま使う場合は初回利用時に `cc` で自動コンパイルを試みます。
OpenMP スレッド内で BLAS が多重にスレッドを立てないよう，`threadpoolctl` で BLAS を
1 スレッドに制限します (`threadpoolctl` は依存パッケージに含めています)。
ブタン/cc-pVDZ (14 原子，4 スレッド) での (T) 分割の時間は，原子ごとに $R^A$ を作る旧実装の
707 秒，numpy 版の 154 秒 / 128 秒 (`with_t4` True / False) に対し，C カーネルでは 17 秒 / 12 秒で，
PySCF の (T) エネルギー本体 (7 秒) の 2 倍前後です。
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
# SCF/CC の収束判定 (既定値): conv_tol=1e-10, conv_tol_grad=1e-7, newton=True,
#                             cc_conv_tol=1e-9, cc_conv_tol_normt=1e-7
print(res.summary())
print(res.e_corr)      # CBS 極限の原子相関エネルギー
print(res.e_tot)       # E_HF^A (参照) + E_corr^{CBS,A}
print(res.corr)        # {(手法, 基底): 各レベルの原子相関エネルギー}
print(res.eda_results) # 各レベルの EDA 結果オブジェクト
```

主なオプション: `basis_family='cc-pv' | 'aug-cc-pv'`，`frozen='auto' | int`，
`orbital_basis='nao' | 'ao' | 'lso'`，`ne_partition`，`w_occ`，`hf_cbs`。

### CBS エネルギーの解析的核座標勾配

`with_grad=True` を指定すると，各レベル (RHF, MP2, CCSD, CCSD(T); frozen core 対応) の解析的勾配を
エネルギー計算に用いたのと同じ SCF/MP2/CCSD オブジェクトから求め，モデルの係数で線形結合して
CBS エネルギーの勾配 $\partial E_{CBS}/\partial \mathbf R$ (`res.grad`, natm × 3, hartree/bohr) を返します
(エネルギー計算の重複はありません)。HF/CBS 部分は `hf_cbs` の方式 (すべてエネルギーについて線形) の
係数で同様に線形結合します。各レベルの勾配は `res.grads[(手法, 基底)]`，`res.hf_grads[基底]` に保存されます。

```python
res = eda_cbs.QTD(mol, with_grad=True).kernel()
print(res.e_tot)     # CCSD(T)/CBS の原子エネルギー
print(res.grad)      # CBS エネルギーの核座標勾配
```

**注意 (PySCF の CCSD(T) 勾配):** `pyscf.grad.ccsd_t.Gradients(mycc).kernel()` を引数なしで
呼ぶと CCSD のラムダ振幅が使われ，有限差分と 6e-4 hartree/bohr 程度ずれた誤った勾配になります。
`pyscf_eda.grad.ccsd_t_gradient` は `pyscf.cc.ccsd_t_lambda` で CCSD(T) のラムダ方程式を解いてから
勾配を評価し，有限差分と 1e-6 で一致することをテストで確認しています。

### Hartree–Fock エネルギーの CBS 見積り

HF 部分はフィッティングモデルの対象外なので，途中で得られる DZ, TZ, QZ の HF エネルギーから
`hf_cbs` で指定した方法で CBS 極限を見積もります。

| `hf_cbs` | 式 | 性質 |
|---|---|---|
| `'halkier'` (既定) | 2 点 (TZ, QZ) $E(X)=E_{CBS}+A\,e^{-\alpha X}$，$\alpha=1.63$ 固定 (`hf_alpha`) | エネルギーについて線形 |
| `'karton-martin'` | 2 点 (TZ, QZ) $E(X)=E_{CBS}+A\,(X+1)e^{-9\sqrt{X}}$ | 線形 |
| `'largest'` | 外挿なし (QZ の値) | – |

いずれもエネルギーについて線形なので，原子ごとに適用しても和が分子の外挿値と一致し
(size-consistent)，勾配も同じ係数の線形結合で得られます。指数もフィットする非線形の
3 点式 (Feller 式など) は意図的に採用していません。原子ごとに適用すると和が分子の値と
一致せず，原子の HF エネルギーが $X$ に対して単調に収束していないとき (例: H2O の H 原子は
TZ→QZ で上昇) には不安定になるためです。
結果には全方式の見積り (原子ごと・分子全体) と原子和と分子値の差 (丸め誤差の検査) が
`res.hf_estimates` に格納され，`summary()` にも表示されます。
さらに，選んだ方式だけでなく全方式の HF/CBS に相関エネルギーの CBS 値を加えた
全エネルギー (原子・分子) を `res.e_tot_by_hf` / `res.e_tot_mol_by_hf` に格納し，
`summary()` の末尾に「E_QTD halkier」「E_QTD karton-martin」… の行として出力します
(勾配計算時は `res.grad_by_hf` に各方式の CBS 勾配も入ります)。HF のみを調べるには
`eda_cbs.HFCBS(mol).kernel()` (または `eda_cbs.hf_cbs(mol)`) を使います。

```python
hfres = eda_cbs.HFCBS(mol, basis_family='cc-pv').kernel()   # RHF/DZ,TZ,QZ + EDA のみ
print(hfres.summary())          # 全方式の原子 HF/CBS，原子和 − 分子値
print(hfres.estimates['halkier']['atoms'])
```

### 収束判定と勾配の精度

解析勾配の誤差は SCF の軌道勾配の残差の 1 次 (エネルギーの誤差は 2 次) なので，
勾配の有効桁数を決めるのは `conv_tol` ではなく軌道勾配ノルムの閾値 `conv_tol_grad` です
(PySCF の既定は √conv_tol ≈ 3e-5 で，勾配の誤差は約 5e-7 hartree/bohr)。
`CompositeEDA` と `HFCBS` は既定で `conv_tol_grad=1e-7` (勾配の誤差 ~1e-8 hartree/bohr，
約 8 桁) を用い，CCSD は `cc_conv_tol=1e-9`, `cc_conv_tol_normt=1e-7` (ラムダ方程式にも
同じ閾値) で解きます。

`newton=True` (既定) では各 SCF を 3 段階で解きます: 短い DIIS の予備収束 → 二次収束
(Newton, `mf.newton()`) 法で軌道勾配 1e-6 まで → 収束した密度から DIIS で目標の閾値まで
仕上げ (通常 1〜2 回)。大きな分子で DIIS が数値ノイズの水準で停滞しても Newton 法は
1e-6 まで確実に収束します (Newton 法の拡張 Hessian は |g| ≲ 1e-7 で精度を失うので，
最後の桁は DIIS に任せます)。`newton=False` で通常の DIIS のみになります。
これらの SCF は `pyscf_eda.cbs.run_scf(mol, conv_tol, conv_tol_grad, newton)` として単独でも
使えます (返り値は通常の `scf.RHF` オブジェクト)。


## 例

* `examples/h2o_rhf_eda.py` – 論文 Table 1 と同じ H2O 構造 (cc-pVDZ) での RHF-EDA。
  孤立原子からの差 (Table 1 の括弧内に相当) は，孤立原子を通常の UHF で計算して求めます
  (単一原子ではその全エネルギーがそのまま原子エネルギーになるため EDA は不要です)。
* `examples/h2o_bond_breaking.py` – O–H 結合伸長 (2002 年論文 Table 2 に対応) に伴う原子エネルギーの変化
* `examples/co2_nao_eda.py` – CO2 (RHF) での従来型 / LSO- / NAO-EDA の基底関数依存性
* `examples/h2o_b3lyp_table1.py` – H2O, B3LYP/cc-pVDZ の EDA (2002 年論文 Table 1 の再現)
* `examples/co2_b3lyp_2006.py` – CO2, B3LYP の各基底での MPA/LPA/NPA と EDA/LSO/NAO-EDA (2006 年論文 Table 1, 2 との比較)
* `examples/g2_mulliken_grid_eda.py` – G2-1 分子の Mulliken-EDA / Grid-EDA / 従来型 EDA (2009 年論文 Table II, III との比較)
* `examples/h2o_mp2_eda.py` – H2O の MP2-EDA (占有側 / 仮想側分割の比較，NAO 基底，UMP2 孤立原子との差)
* `examples/h2o_ccsd_eda.py` – H2O の CCSD-EDA (MP2-EDA との比較，NAO 基底，UCCSD 孤立原子との差)
* `examples/h2o_ccsd_t_eda.py` – H2O の CCSD(T)-EDA ($U^{0,0}$ / $U^{2,2}$，$E_T^{[4]}$ / $E_{ST}^{[5]}$，UCCSD(T) 孤立原子との差)
* `examples/h2o_cbs_eda.py` – H2O の QDD / QTD による CBS 極限の原子エネルギー
* `examples/h2o_hf_cbs.py` – H2O の原子 HF エネルギーの CBS 見積り (線形 / 非線形外挿の安定性の検証)
* `examples/h2o_cbs_energy_gradient.py` – H2O の CCSD(T)/CBS 原子エネルギー (QDD, QTD) と CBS エネルギーの解析的勾配を 1 つの入力で計算

## 制限事項

* 現在は閉殻 RHF / RKS，閉殻 MP2，閉殻 CCSD，閉殻 CCSD(T) のみ対応しています (UHF/ROHF/UKS,
  UMP2, UCCSD は未対応で例外を出します。MP2/CC の EDA は HF 参照のみ)。検証などで孤立原子のエネルギーが必要な場合は，
  通常の UHF / UMP2 / UCCSD(T) 計算を用いてください。
* MP2-/CCSD-EDA は原子ごとに半変換積分 $(la|jb)$, $(il|jb)$ を作るため，メモリは
  $O(n_A n_{vir} n_{occ} n_{vir})$，計算量は原子数 × 積分変換のコストになります。
  CCSD-EDA は CCSD の相関エネルギーのみを分割し，(T) 補正は `pyscf_eda.ccsd_t` で扱います。
  密度フィッティング (`.density_fit()`) や ECP を用いた RHF には対応しています。
* `mf.get_hcore()` に $T + V_{nuc} (+V_{ECP})$ 以外の一電子項 (外場など) が含まれる場合，
  その寄与は基底関数で分割し `e_other` として報告します。

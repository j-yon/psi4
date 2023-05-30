/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2022 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with Psi4; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#include "dlpno.h"
#include "sparse.h"

#include "psi4/lib3index/3index.h"
#include "psi4/libdiis/diismanager.h"
#include "psi4/libfock/cubature.h"
#include "psi4/libfock/points.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/local.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/orthog.h"
#include "psi4/libmints/twobody.h"
#include "psi4/libmints/vector.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libqt/qt.h"

#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace psi {
namespace dlpno {

DLPNOCCSD_T::DLPNOCCSD_T(SharedWavefunction ref_wfn, Options& options) : DLPNOCCSD(ref_wfn, options) {}
DLPNOCCSD_T::~DLPNOCCSD_T() {}

void DLPNOCCSD_T::tno_transform() {
    timer_on("TNO transform");

    int naocc = nalpha_ - nfrzc();
    int n_lmo_pairs = ij_to_i_j_.size();
    int npao = C_pao_->colspi(0);

    int ijk = 0;
    for (int ij = 0; ij < n_lmo_pairs; ij++) {
        int i, j;
        std::tie(i, j) = ij_to_i_j_[ij];
        for (int k : lmopair_to_lmos_[ij]) {
            ijk_to_i_j_k_.push_back(std::make_tuple(i, j, k));
            i_j_k_to_ijk_[i * naocc * naocc + j * naocc + k] = ijk;
            ijk++;
        }
    }

    int n_lmo_triplets = ijk_to_i_j_k_.size();
    lmotriplet_to_paos_.resize(n_lmo_triplets);
    X_tno_.resize(n_lmo_triplets);
    e_tno_.resize(n_lmo_triplets);
    n_tno_.resize(n_lmo_triplets);
    denom_ijk_.resize(n_lmo_triplets);

    std::vector<SharedMatrix> D_ij(n_lmo_pairs);

#pragma omp parallel for schedule(dynamic, 1)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        int i, j;
        std::tie(i, j) = ij_to_i_j_[ij];

        if (i > j || n_pno_[ij] == 0) continue;

        D_ij[ij] = linalg::doublet(Tt_iajb_[ij], T_iajb_[ij], false, true);
        D_ij[ij]->add(linalg::doublet(Tt_iajb_[ij], T_iajb_[ij], true, false));

        D_ij[ij] = linalg::triplet(X_pno_[ij], D_ij[ij], X_pno_[ij], false, false, true);

        if (i < j) {
            int ji = ij_to_ji_[ij];
            D_ij[ji] = D_ij[ij]->clone();
        }
    }

    std::vector<std::vector<int>> global_pao_to_pao_ij(n_lmo_pairs);

#pragma omp parallel for schedule(dynamic, 1)
    for (int ij = 0; ij < n_lmo_pairs; ++ij) {
        global_pao_to_pao_ij[ij] = std::vector<int>(npao, -1);

        for (int u_ij = 0; u_ij < lmopair_to_paos_[ij].size(); u_ij++) {
            int u = lmopair_to_paos_[ij][u_ij];
            global_pao_to_pao_ij[ij][u] = u_ij;
        }
    }

#pragma omp parallel for schedule(static, 1)
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        int i, j, k;
        std::tie(i, j, k) = ijk_to_i_j_k_[ijk];
        int ij = i_j_to_ij_[i][j], jk = i_j_to_ij_[j][k], ik = i_j_to_ij_[i][k];

        lmotriplet_to_paos_[ijk] = merge_lists(lmopair_to_paos_[ij], lmo_to_paos_[k]);

        if (i > j || i > k || j > k) continue;

        // number of PAOs in the triplet domain (before removing linear dependencies)
        int npao_ijk = lmotriplet_to_paos_[ijk].size();

        // Form the triplet density (from pair densities in redundant basis)
        auto D_ijk = std::make_shared<Matrix>("D_ijk", npao_ijk, npao_ijk);
        D_ijk->zero();

        for (int u_ijk = 0; u_ijk < lmotriplet_to_paos_[ijk].size(); u_ijk++) {
            int u = lmotriplet_to_paos_[ijk][u_ijk];
            int u_ij = global_pao_to_pao_ij[ij][u], u_jk = global_pao_to_pao_ij[jk][u], u_ik = global_pao_to_pao_ij[ik][u];
            
            for (int v_ijk = 0; v_ijk < lmotriplet_to_paos_[ijk].size(); v_ijk++) {
                int v = lmotriplet_to_paos_[ijk][v_ijk];
                int v_ij = global_pao_to_pao_ij[ij][v], v_jk = global_pao_to_pao_ij[jk][v], v_ik = global_pao_to_pao_ij[ik][v];

                if (n_pno_[ij] > 0 && u_ij != -1 && v_ij != -1) (*D_ijk)(u_ijk, v_ijk) += (*D_ij[ij])(u_ij, v_ij);
                if (n_pno_[jk] > 0 && u_jk != -1 && v_jk != -1) (*D_ijk)(u_ijk, v_ijk) += (*D_ij[jk])(u_jk, v_jk);
                if (n_pno_[ik] > 0 && u_ik != -1 && v_ik != -1) (*D_ijk)(u_ijk, v_ijk) += (*D_ij[ik])(u_ik, v_ik);
            }
        }
        D_ijk->scale(1.0 / 3.0);

        // Canonicalize PAOs of triplet ijk
        auto S_pao_ijk = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmotriplet_to_paos_[ijk]);
        auto F_pao_ijk = submatrix_rows_and_cols(*F_pao_, lmotriplet_to_paos_[ijk], lmotriplet_to_paos_[ijk]);

        SharedMatrix X_pao_ijk;
        SharedVector e_pao_ijk;
        std::tie(X_pao_ijk, e_pao_ijk) = orthocanonicalizer(S_pao_ijk, F_pao_ijk);

        F_pao_ijk = linalg::triplet(X_pao_ijk, F_pao_ijk, X_pao_ijk, true, false, false);
        D_ijk = linalg::triplet(X_pao_ijk, D_ijk, X_pao_ijk, true, false, false);

        size_t nvir_ijk = F_pao_ijk->rowspi(0);

        // Diagonalization of triplet density gives TNOs (in basis of LMO's virtual domain)
        // as well as TNO occ numbers
        auto X_tno_ijk = std::make_shared<Matrix>("eigenvectors", nvir_ijk, nvir_ijk);
        Vector tno_occ("eigenvalues", nvir_ijk);
        D_ijk->diagonalize(*X_tno_ijk, tno_occ, descending);

        int nvir_ijk_final = 0;
        for (size_t a = 0; a < nvir_ijk; ++a) {
            if (fabs(tno_occ.get(a)) >= T_CUT_TNO_) {
                nvir_ijk_final++;
            }
        }

        Dimension zero(1);
        Dimension dim_final(1);
        dim_final.fill(nvir_ijk_final);

        // This transformation gives orbitals that are orthonormal but not canonical
        X_tno_ijk = X_tno_ijk->get_block({zero, X_tno_ijk->rowspi()}, {zero, dim_final});
        tno_occ = tno_occ.get_block({zero, dim_final});

        SharedMatrix tno_canon;
        SharedVector e_tno_ijk;
        std::tie(tno_canon, e_tno_ijk) = canonicalizer(X_tno_ijk, F_pao_ijk);

        X_tno_ijk = linalg::doublet(X_tno_ijk, tno_canon, false, false);
        X_tno_ijk = linalg::doublet(X_pao_ijk, X_tno_ijk, false, false);

        X_tno_[ijk] = X_tno_ijk;
        e_tno_[ijk] = e_tno_ijk;
        n_tno_[ijk] = X_tno_ijk->colspi(0);

        denom_ijk_[ijk] = std::make_shared<Matrix>(n_tno_[ijk], n_tno_[ijk] * n_tno_[ijk]);
        for (int a_ijk = 0; a_ijk < n_tno_[ijk]; a_ijk++) {
            for (int b_ijk = 0; b_ijk < n_tno_[ijk]; b_ijk++) {
                for (int c_ijk = 0; c_ijk < n_tno_[ijk]; c_ijk++) {
                    (*denom_ijk_[ijk])(a_ijk, b_ijk * n_tno_[ijk] + c_ijk) = (*e_tno_[ijk])(a_ijk) + (*e_tno_[ijk])(b_ijk) 
                        + (*e_tno_[ijk])(c_ijk) - (*F_lmo_)(i,i) - (*F_lmo_)(j,j) - (*F_lmo_)(k,k);
                }
            }
        }

        // account for symmetry
        if (i != j || j != k || i != k) {
            int ikj = i_j_k_to_ijk_[i * naocc * naocc + k * naocc + j];
            int jik = i_j_k_to_ijk_[j * naocc * naocc + i * naocc + k];
            int jki = i_j_k_to_ijk_[j * naocc * naocc + k * naocc + i];
            int kij = i_j_k_to_ijk_[k * naocc * naocc + i * naocc + j];
            int kji = i_j_k_to_ijk_[k * naocc * naocc + j * naocc + i];

            std::vector<int> perms{ikj, jik, jki, kij, kji};
            for (int perm : perms) {
                X_tno_[perm] = X_tno_ijk;
                e_tno_[perm] = e_tno_ijk;
                n_tno_[perm] = X_tno_ijk->colspi(0);
                denom_ijk_[perm] = denom_ijk_[ijk];
            }
        }
    }

    int tno_count_total = 0, tno_count_min = C_pao_->colspi(0), tno_count_max = 0;
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        tno_count_total += n_tno_[ijk];
        tno_count_min = std::min(tno_count_min, n_tno_[ijk]);
        tno_count_max = std::max(tno_count_max, n_tno_[ijk]);
    }

    outfile->Printf("  \n");
    outfile->Printf("    Number of Local MO triplets: %d\n", n_lmo_triplets);
    outfile->Printf("    Max Number of Possible LMO Triplets: %d (Ratio: %.4f)\n", naocc * naocc * naocc,
    (double) n_lmo_triplets / (naocc * naocc * naocc));
    outfile->Printf("    Natural Orbitals per Local MO triplet:\n");
    outfile->Printf("      Avg: %3d NOs \n", tno_count_total / n_lmo_triplets);
    outfile->Printf("      Min: %3d NOs \n", tno_count_min);
    outfile->Printf("      Max: %3d NOs \n", tno_count_max);
    outfile->Printf("  \n");

    timer_off("TNO transform");
}

void DLPNOCCSD_T::compute_tno_overlaps() {

    timer_on("TNO overlaps");

    int n_lmo_triplets = ijk_to_i_j_k_.size();
    int naocc = nalpha_ - nfrzc();

    S_ijk_ii_.resize(n_lmo_triplets);
    S_ijk_ij_.resize(n_lmo_triplets);
    S_ijk_il_.resize(n_lmo_triplets);
    S_ijk_ljk_.resize(n_lmo_triplets);

#pragma omp parallel for schedule(dynamic, 1)
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        int i, j, k;
        std::tie(i, j, k) = ijk_to_i_j_k_[ijk];

        int ntno_ijk = n_tno_[ijk];
        if (ntno_ijk == 0) continue;

        // Compute overlap between TNOs of triplet ijk and PNOs of pair ii
        int ii = i_j_to_ij_[i][i];
        auto S_pao_ijk_ii = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmopair_to_paos_[ii]);
        S_ijk_ii_[ijk] = linalg::triplet(X_tno_[ijk], S_pao_ijk_ii, X_pno_[ii], true, false, false);

        // Compute overlap between TNOs of triplet ijk and PNOs of pair ij
        int ij = i_j_to_ij_[i][j];
        if (n_pno_[ij] > 0) {
            auto S_pao_ijk_ij = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmopair_to_paos_[ij]);
            S_ijk_ij_[ijk] = linalg::triplet(X_tno_[ijk], S_pao_ijk_ij, X_pno_[ij], true, false, false);
        }

        // Compute overlap between TNOs of triplet ijk and PNOs of pair ij
        S_ijk_il_[ijk].resize(naocc);
        for (int l = 0; l < naocc; l++) {
            int jk = i_j_to_ij_[j][k];
            int l_jk = lmopair_to_lmos_dense_[jk][l];

            int il = i_j_to_ij_[i][l];

            if (l_jk != -1 && il != -1 && n_pno_[il] != 0) {
                auto S_pao_ijk_il = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmopair_to_paos_[il]);
                S_ijk_il_[ijk][l] = linalg::triplet(X_tno_[ijk], S_pao_ijk_il, X_pno_[il], true, false, false);
            }
        }

        // Compute TNO overlaps between triplet spaces ijk and ljk
        S_ijk_ljk_[ijk].resize(naocc);

        for (int l = 0; l < naocc; l++) {
            int ljk_dense = l * naocc * naocc + j * naocc + k;
            if (!i_j_k_to_ijk_.count(ljk_dense)) continue;

            int ljk = i_j_k_to_ijk_[ljk_dense];
            if (n_tno_[ljk] == 0) continue;

            auto S_pao_ijk_ljk = submatrix_rows_and_cols(*S_pao_, lmotriplet_to_paos_[ijk], lmotriplet_to_paos_[ljk]);
            S_ijk_ljk_[ijk][l] = linalg::triplet(X_tno_[ijk], S_pao_ijk_ljk, X_tno_[ljk], true, false, false);
        }
    }

    timer_off("TNO overlaps");
}

void DLPNOCCSD_T::compute_W_iajbkc() {
    timer_on("Compute W_iajbkc");

    int naocc = nalpha_ - nfrzc();
    int n_lmo_triplets = ijk_to_i_j_k_.size();

    std::vector<SharedMatrix> W_temp(n_lmo_triplets);

#pragma omp parallel for schedule(dynamic)
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        int i, j, k;
        std::tie(i, j, k) = ijk_to_i_j_k_[ijk];

        int ii = i_j_to_ij_[i][i];
        int ij = i_j_to_ij_[i][j], jk = i_j_to_ij_[j][k], ik = i_j_to_ij_[i][k];
        int kj = ij_to_ji_[jk];

        int ntno_ijk = n_tno_[ijk];

        if (ntno_ijk == 0) continue;
        
        W_temp[ijk] = std::make_shared<Matrix>(ntno_ijk, ntno_ijk * ntno_ijk);

        if (n_pno_[kj] > 0 && n_pno_[ij] > 0) {
            auto T_kj = linalg::doublet(T_iajb_[kj], S_PNO(kj, ij));
            auto K_temp1 = linalg::doublet(T_kj, K_tilde_phys_[ij]);
            int kji = i_j_k_to_ijk_[k * naocc * naocc + j * naocc + i];
            K_temp1 = linalg::doublet(S_ijk_ij_[kji], K_temp1, false, false);

            for (int c_ijk = 0; c_ijk < ntno_ijk; c_ijk++) {
                std::vector<int> c_ijk_slice(1, c_ijk);
                auto K_temp1a = submatrix_rows(*K_temp1, c_ijk_slice);
                K_temp1a->reshape(n_pno_[ij], n_pno_[ij]);

                K_temp1a = linalg::triplet(S_ijk_ij_[ijk], K_temp1a, S_ijk_ij_[ijk], false, false, true);

                for (int a_ijk = 0; a_ijk < ntno_ijk; a_ijk++) {
                    for (int b_ijk = 0; b_ijk < ntno_ijk; b_ijk++) {
                        (*W_temp[ijk])(a_ijk, b_ijk * ntno_ijk + c_ijk) = (*K_temp1a)(a_ijk, b_ijk);
                    }
                }
            }
        }

        if (n_pno_[kj] > 0) {
            int jki = i_j_k_to_ijk_[j * naocc * naocc + k * naocc + i];
            auto K_temp2 = linalg::doublet(K_bar_[jk], S_ijk_ij_[jki], false, true);

            for (int l_jk = 0; l_jk < lmopair_to_lmos_[jk].size(); l_jk++) {
                int l = lmopair_to_lmos_[jk][l_jk];
                int il = i_j_to_ij_[i][l];

                if (il == -1 || n_pno_[il] == 0) continue;

                auto T_il = linalg::triplet(S_ijk_il_[ijk][l], T_iajb_[il], S_ijk_il_[ijk][l], false, false, true);

                for (int a_ijk = 0; a_ijk < ntno_ijk; a_ijk++) {
                    for (int b_ijk = 0; b_ijk < ntno_ijk; b_ijk++) {
                        for (int c_ijk = 0; c_ijk < ntno_ijk; c_ijk++) {
                            (*W_temp[ijk])(a_ijk, b_ijk * ntno_ijk + c_ijk) -= (*T_il)(a_ijk, b_ijk) * (*K_temp2)(l_jk, c_ijk);
                        }
                    }
                }
            }
        }
        
    }

    W_iajbkc_.resize(n_lmo_triplets);
#pragma omp parallel for schedule(dynamic)
    for (int ijk = 0; ijk < n_lmo_triplets; ijk++) {
        int i, j, k;
        std::tie(i, j, k) = ijk_to_i_j_k_[ijk];

        int ntno_ijk = n_tno_[ijk];
        if (ntno_ijk == 0) continue;

        W_iajbkc_[ijk] = std::make_shared<Matrix>(ntno_ijk, ntno_ijk * ntno_ijk);

        int ikj = i_j_k_to_ijk_[i * naocc * naocc + k * naocc + j];
        int jik = i_j_k_to_ijk_[j * naocc * naocc + i * naocc + k];
        int jki = i_j_k_to_ijk_[j * naocc * naocc + k * naocc + i];
        int kij = i_j_k_to_ijk_[k * naocc * naocc + i * naocc + j];
        int kji = i_j_k_to_ijk_[k * naocc * naocc + j * naocc + i];

        for (int a_ijk = 0; a_ijk < ntno_ijk; a_ijk++) {
            for (int b_ijk = 0; b_ijk < ntno_ijk; b_ijk++) {
                for (int c_ijk = 0; c_ijk < ntno_ijk; c_ijk++) {
                    (*W_iajbkc_[ijk])(a_ijk, b_ijk * ntno_ijk + c_ijk) = (*W_temp[ijk])(a_ijk, b_ijk * ntno_ijk + c_ijk) +
                        (*W_temp[ikj])(a_ijk, c_ijk * ntno_ijk + b_ijk) + (*W_temp[jik])(b_ijk, a_ijk * ntno_ijk + c_ijk) +
                        (*W_temp[jki])(b_ijk, c_ijk * ntno_ijk + a_ijk) + (*W_temp[kij])(c_ijk, a_ijk * ntno_ijk + b_ijk) + 
                        (*W_temp[kji])(c_ijk, b_ijk * ntno_ijk + a_ijk);
                }
            }
        }

    }

    timer_off("Compute W_iajbkc");
}

double DLPNOCCSD_T::compute_t_energy() {
    timer_on("Compute (T) Energy");

    int naocc = nalpha_ - nfrzc();
    int n_lmo_triplets = ijk_to_i_j_k_.size();

    double E_T = 0.0;

    std::vector<SharedMatrix> V_ijk(n_lmo_triplets);
#pragma omp parallel for schedule(dynamic)
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        int i, j, k;
        std::tie(i, j, k) = ijk_to_i_j_k_[ijk];

        int ntno_ijk = n_tno_[ijk];
        V_ijk[ijk] = std::make_shared<Matrix>("V_ijk", ntno_ijk, ntno_ijk * ntno_ijk);
    }

#pragma omp parallel for schedule(dynamic)
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        int i, j, k;
        std::tie(i, j, k) = ijk_to_i_j_k_[ijk];

        int ij = i_j_to_ij_[i][j], jk = i_j_to_ij_[j][k], ik = i_j_to_ij_[i][k];
        int ii = i_j_to_ij_[i][i], jj = i_j_to_ij_[j][j], kk = i_j_to_ij_[k][k];

        if (i > j || i > k || j > k) continue;

        int ntno_ijk = n_tno_[ijk];
        if (ntno_ijk > 0) {
            int jki = i_j_k_to_ijk_[j * naocc * naocc + k * naocc + i];
            int ikj = i_j_k_to_ijk_[i * naocc * naocc + k * naocc + j];
            int kij = i_j_k_to_ijk_[k * naocc * naocc + i * naocc + j];

            V_ijk[ijk]->copy(W_iajbkc_[ijk]);

            SharedMatrix T_temp1, K_temp1, T_temp2, K_temp2, T_temp3, K_temp3;

            if (n_pno_[jk] > 0) {
                T_temp1 = linalg::doublet(S_ijk_ii_[ijk], T_ia_[i], false, false);
                K_temp1 = linalg::triplet(S_ijk_ij_[jki], K_iajb_[jk], S_ijk_ij_[jki], false, false, true);
            }

            if (n_pno_[ik] > 0) {
                T_temp2 = linalg::doublet(S_ijk_ii_[jki], T_ia_[j], false, false);
                K_temp2 = linalg::triplet(S_ijk_ij_[ikj], K_iajb_[ik], S_ijk_ij_[ikj], false, false, true);
            }

            if (n_pno_[ij] > 0) {
                T_temp3 = linalg::doublet(S_ijk_ii_[kij], T_ia_[k], false, false);
                K_temp3 = linalg::triplet(S_ijk_ij_[ijk], K_iajb_[ij], S_ijk_ij_[ijk], false, false, true);
            }

            for (int a_ijk = 0; a_ijk < ntno_ijk; a_ijk++) {
                for (int b_ijk = 0; b_ijk < ntno_ijk; b_ijk++) {
                    for (int c_ijk = 0; c_ijk < ntno_ijk; c_ijk++) {
                        double cont = 0.0;
                        if (n_pno_[jk] > 0) cont += (*T_temp1)(a_ijk, 0) * (*K_temp1)(b_ijk, c_ijk);
                        if (n_pno_[ik] > 0) cont += (*T_temp2)(b_ijk, 0) * (*K_temp2)(a_ijk, c_ijk);
                        if (n_pno_[ij] > 0) cont += (*T_temp3)(c_ijk, 0) * (*K_temp3)(a_ijk, b_ijk);
                        (*V_ijk[ijk])(a_ijk, b_ijk * ntno_ijk + c_ijk) += cont;
                    }
                }
            }
        } // end if

        t_symmetrizer(V_ijk, ijk);
        
    }

#pragma omp parallel for schedule(dynamic) reduction(+ : E_T)
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        int i, j, k;
        std::tie(i, j, k) = ijk_to_i_j_k_[ijk];

        if (i > j || i > k || j > k) continue;

        int kji = i_j_k_to_ijk_[k * naocc * naocc + j * naocc + i];
        int ikj = i_j_k_to_ijk_[i * naocc * naocc + k * naocc + j];
        int jik = i_j_k_to_ijk_[j * naocc * naocc + i * naocc + k];
        int jki = i_j_k_to_ijk_[j * naocc * naocc + k * naocc + i];
        int kij = i_j_k_to_ijk_[k * naocc * naocc + i * naocc + j];

        double prefactor = 1.0;
        if (i == j && j == k) {
            prefactor /= 6.0;
        } else if (i == j || j == k || i == k) {
            prefactor /= 2.0;
        }

        E_T += 8.0 * prefactor * V_ijk[ijk]->vector_dot(T_iajbkc_[ijk]);
        E_T -= 4.0 * prefactor * V_ijk[kji]->vector_dot(T_iajbkc_[ijk]);
        E_T -= 4.0 * prefactor * V_ijk[ikj]->vector_dot(T_iajbkc_[ijk]);
        E_T -= 4.0 * prefactor * V_ijk[jik]->vector_dot(T_iajbkc_[ijk]);
        E_T += 2.0 * prefactor * V_ijk[jki]->vector_dot(T_iajbkc_[ijk]);
        E_T += 2.0 * prefactor * V_ijk[kij]->vector_dot(T_iajbkc_[ijk]);
    }

    timer_off("Compute (T) Energy");

    return E_T;
}

SharedMatrix matmul_3d(SharedMatrix A, SharedMatrix X, int dim_old, int dim_new) {
    /*
    Performs the operation A'[i,j,k] = A[I,J,K] * X[i,I] * X[j,J] * X[k,K] for cube 3d tensors
    */
    
    SharedMatrix A_new = linalg::doublet(X, A, false, false);
    A_new->reshape(dim_new * dim_old, dim_old);
    A_new = linalg::doublet(A_new, X, false, true);

    SharedMatrix A_T = std::make_shared<Matrix>(dim_new * dim_new, dim_old);
    for (int ind = 0; ind < dim_new * dim_new * dim_old; ++ind) {
        int a = ind / (dim_new * dim_old), b = (ind / dim_old) % dim_new, c = ind % dim_old;
        (*A_T)(a * dim_new + b, c) = (*A_new)(a * dim_old + c, b);
    }
    A_T = linalg::doublet(A_T, X, false, true);

    A_new = std::make_shared<Matrix>(dim_new, dim_new * dim_new);

    for (int ind = 0; ind < dim_new * dim_new * dim_new; ++ind) {
        int a = ind / (dim_new * dim_new), b = (ind / dim_new) % dim_new, c = ind % dim_new;
        (*A_new)(a, b * dim_new + c) = (*A_T)(a * dim_new + c, b);
    }

    return A_new;

}

inline void DLPNOCCSD_T::t_symmetrizer(std::vector<SharedMatrix>& X, int ijk) {
    int i, j, k;
    std::tie(i, j, k) = ijk_to_i_j_k_[ijk];

    int naocc = nalpha_ - nfrzc();
    int ntno_ijk = n_tno_[ijk];

    if (i == j && j == k) {
        return;
    } else if (i == j) {
        // iik
        int iki = i_j_k_to_ijk_[i * naocc * naocc + k * naocc + i];
        int kii = i_j_k_to_ijk_[k * naocc * naocc + i * naocc + i];

        std::vector<int> perms{iki, kii};

        for (int a_ijk = 0; a_ijk < ntno_ijk; a_ijk++) {
            for (int b_ijk = 0; b_ijk < ntno_ijk; b_ijk++) {
                for (int c_ijk = 0; c_ijk < ntno_ijk; c_ijk++) {
                    double val = (*X[ijk])(a_ijk, b_ijk * ntno_ijk + c_ijk);
                    (*X[iki])(a_ijk, c_ijk * ntno_ijk + b_ijk) = val;
                    (*X[kii])(c_ijk, a_ijk * ntno_ijk + b_ijk) = val;
                }
            }
        }
    } else if (i == k) {
        // iji
        int iij = i_j_k_to_ijk_[i * naocc * naocc + i * naocc + j];
        int jii = i_j_k_to_ijk_[j * naocc * naocc + i * naocc + i];

        std::vector<int> perms{iij, jii};

        for (int a_ijk = 0; a_ijk < ntno_ijk; a_ijk++) {
            for (int b_ijk = 0; b_ijk < ntno_ijk; b_ijk++) {
                for (int c_ijk = 0; c_ijk < ntno_ijk; c_ijk++) {
                    double val = (*X[ijk])(a_ijk, b_ijk * ntno_ijk + c_ijk);
                    (*X[iij])(a_ijk, c_ijk * ntno_ijk + b_ijk) = val;
                    (*X[jii])(b_ijk, a_ijk * ntno_ijk + c_ijk) = val;
                }
            }
        }
    } else if (j == k) {
        // ijj
        int jij = i_j_k_to_ijk_[j * naocc * naocc + i * naocc + j];
        int jji = i_j_k_to_ijk_[j * naocc * naocc + j * naocc + i];

        std::vector<int> perms{jij, jji};

        for (int a_ijk = 0; a_ijk < ntno_ijk; a_ijk++) {
            for (int b_ijk = 0; b_ijk < ntno_ijk; b_ijk++) {
                for (int c_ijk = 0; c_ijk < ntno_ijk; c_ijk++) {
                    double val = (*X[ijk])(a_ijk, b_ijk * ntno_ijk + c_ijk);
                    (*X[jij])(b_ijk, a_ijk * ntno_ijk + c_ijk) = val;
                    (*X[jji])(c_ijk, b_ijk * ntno_ijk + a_ijk) = val;
                }
            }
        }
    } else {
        int ikj = i_j_k_to_ijk_[i * naocc * naocc + k * naocc + j];
        int jik = i_j_k_to_ijk_[j * naocc * naocc + i * naocc + k];
        int jki = i_j_k_to_ijk_[j * naocc * naocc + k * naocc + i];
        int kij = i_j_k_to_ijk_[k * naocc * naocc + i * naocc + j];
        int kji = i_j_k_to_ijk_[k * naocc * naocc + j * naocc + i];

        std::vector<int> perms{ikj, jik, jki, kij, kji};

        for (int a_ijk = 0; a_ijk < ntno_ijk; a_ijk++) {
            for (int b_ijk = 0; b_ijk < ntno_ijk; b_ijk++) {
                for (int c_ijk = 0; c_ijk < ntno_ijk; c_ijk++) {
                    double val = (*X[ijk])(a_ijk, b_ijk * ntno_ijk + c_ijk);
                    (*X[ikj])(a_ijk, c_ijk * ntno_ijk + b_ijk) = val;
                    (*X[jik])(b_ijk, a_ijk * ntno_ijk + c_ijk) = val;
                    (*X[jki])(b_ijk, c_ijk * ntno_ijk + a_ijk) = val;
                    (*X[kij])(c_ijk, a_ijk * ntno_ijk + b_ijk) = val;
                    (*X[kji])(c_ijk, b_ijk * ntno_ijk + a_ijk) = val;
                }
            }
        }
    }
}

void DLPNOCCSD_T::compute_lccsd_t0() {
    timer_on("LCCSD(T0)");

    outfile->Printf("\n  ==> Computing DLPNO-CCSD(T0) <==\n\n");

    int naocc = nalpha_ - nfrzc();
    int n_lmo_triplets = ijk_to_i_j_k_.size();
    T_iajbkc_.resize(n_lmo_triplets);

    // Allocate Memory for Triples Amplitudes
#pragma omp parallel for schedule(dynamic)
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        int ntno_ijk = n_tno_[ijk];
        T_iajbkc_[ijk] = std::make_shared<Matrix>("T_ijk", ntno_ijk, ntno_ijk * ntno_ijk);
    }

#pragma omp parallel for schedule(dynamic)
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        int i, j, k;
        std::tie(i, j, k) = ijk_to_i_j_k_[ijk];

        if (i > j || i > k || j > k) continue;

        int ntno_ijk = n_tno_[ijk];
        if (ntno_ijk > 0) {
            T_iajbkc_[ijk]->copy(W_iajbkc_[ijk]);

            double *Tbuff = &(*T_iajbkc_[ijk])(0,0);
            double *Dbuff = &(*denom_ijk_[ijk])(0,0);
            for (int vp = 0; vp < ntno_ijk * ntno_ijk * ntno_ijk; ++vp) {
                (*Tbuff) = -(*Tbuff) / (*Dbuff);
                ++Tbuff, ++Dbuff;
            }
        }

        // account for symmetry
        t_symmetrizer(T_iajbkc_, ijk);

    }

    e_lccsd_t_ = e_lccsd_ + compute_t_energy();

    outfile->Printf("\n");
    outfile->Printf("  * DLPNO-CCSD(T0) Correlation Energy: %16.12f \n", e_lccsd_t_);
    outfile->Printf("  * DLPNO-(T0) Contribution:           %16.12f \n\n", e_lccsd_t_ - e_lccsd_);

    timer_off("LCCSD(T0)");
}

void DLPNOCCSD_T::lccsd_t_iterations() {
    timer_on("LCCSD(T) Iterations");

    int naocc = nalpha_ - nfrzc();
    int n_lmo_triplets = ijk_to_i_j_k_.size();

    outfile->Printf("\n  ==> Local CCSD(T) <==\n\n");
    outfile->Printf("    E_CONVERGENCE = %.2e\n", options_.get_double("E_CONVERGENCE"));
    outfile->Printf("    R_CONVERGENCE = %.2e\n\n", options_.get_double("R_CONVERGENCE"));
    outfile->Printf("                         Corr. Energy    Delta E     Max R\n");

    // => Initialize Triples Residuals <= //
    std::vector<SharedMatrix> R_iajbkc(n_lmo_triplets);

#pragma omp parallel for schedule(dynamic)
    for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
        int ntno_ijk = n_tno_[ijk];
        R_iajbkc[ijk] = std::make_shared<Matrix>("R_ijk", ntno_ijk, ntno_ijk * ntno_ijk);
    }

    int iteration = 1, max_iteration = options_.get_int("DLPNO_MAXITER");
    double e_curr = 0.0, e_prev = 0.0, r_curr = 0.0;
    bool e_converged = false, r_converged = false;

    DIISManager diis(options_.get_int("DIIS_MAX_VECS"), "LCCSD(T) DIIS", DIISManager::RemovalPolicy::LargestError, DIISManager::StoragePolicy::InCore);

    while (!(e_converged && r_converged)) {
        // RMS of residual per single LMO, for assesing convergence
        std::vector<double> R_iajbkc_rms(n_lmo_triplets, 0.0);

#pragma omp parallel for schedule(dynamic)
        for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
            int i, j, k;
            std::tie(i, j, k) = ijk_to_i_j_k_[ijk];

            if (i > j || i > k || j > k) continue;

            int ntno_ijk = n_tno_[ijk];

            if (ntno_ijk > 0) {
                R_iajbkc[ijk]->copy(W_iajbkc_[ijk]);

                double *Rbuff = &(*R_iajbkc[ijk])(0,0);
                double *Tbuff = &(*T_iajbkc_[ijk])(0,0);
                double *Dbuff = &(*denom_ijk_[ijk])(0,0);
                for (int vp = 0; vp < ntno_ijk * ntno_ijk * ntno_ijk; ++vp) {
                    (*Rbuff) += (*Tbuff) * (*Dbuff);
                    ++Rbuff, ++Tbuff, ++Dbuff;
                }

                int kij = i_j_k_to_ijk_[k * naocc * naocc + i * naocc + j];
                int jik = i_j_k_to_ijk_[j * naocc * naocc + i * naocc + k];

                for (int l = 0; l < naocc; l++) {
                    int ijl_dense = i * naocc * naocc + j * naocc + l;
                    if (l != k && i_j_k_to_ijk_.count(ijl_dense)) {
                        int ijl = i_j_k_to_ijk_[ijl_dense];
                        if (n_tno_[ijl] == 0) continue;

                        auto T_temp1 = matmul_3d(T_iajbkc_[ijl], S_ijk_ljk_[kij][l], n_tno_[ijl], n_tno_[ijk]);
                        C_DAXPY(ntno_ijk * ntno_ijk * ntno_ijk, -(*F_lmo_)(l,k), &(*T_temp1)(0,0), 1, &(*R_iajbkc[ijk])(0, 0), 1);
                    }

                    int ilk_dense = i * naocc * naocc + l * naocc + k;
                    if (l != j && i_j_k_to_ijk_.count(ilk_dense)) {
                        int ilk = i_j_k_to_ijk_[ilk_dense];
                        if (n_tno_[ilk] == 0) continue;

                        auto T_temp1 = matmul_3d(T_iajbkc_[ilk], S_ijk_ljk_[jik][l], n_tno_[ilk], n_tno_[ijk]);
                        C_DAXPY(ntno_ijk * ntno_ijk * ntno_ijk, -(*F_lmo_)(l,j), &(*T_temp1)(0,0), 1, &(*R_iajbkc[ijk])(0, 0), 1);
                    }

                    int ljk_dense = l * naocc * naocc + j * naocc + k;
                    if (l != i && i_j_k_to_ijk_.count(ljk_dense)) {
                        int ljk = i_j_k_to_ijk_[ljk_dense];
                        if (n_tno_[ljk] == 0) continue;

                        auto T_temp1 = matmul_3d(T_iajbkc_[ljk], S_ijk_ljk_[ijk][l], n_tno_[ljk], n_tno_[ijk]);
                        C_DAXPY(ntno_ijk * ntno_ijk * ntno_ijk, -(*F_lmo_)(l,i), &(*T_temp1)(0,0), 1, &(*R_iajbkc[ijk])(0, 0), 1);
                    }
                }
            }

            // account for symmetry
            t_symmetrizer(R_iajbkc, ijk);
            R_iajbkc_rms[ijk] = R_iajbkc[ijk]->rms();
        }

        // => Update T3 Amplitudes <= //
#pragma omp for schedule(dynamic)
        for (int ijk = 0; ijk < n_lmo_triplets; ++ijk) {
            int i, j, k;
            std::tie(i, j, k) = ijk_to_i_j_k_[ijk];

            if (i > j || i > k || j > k) continue;

            int ntno_ijk = n_tno_[ijk];
            if (ntno_ijk == 0) continue;

            double *Tbuff = &(*T_iajbkc_[ijk])(0,0);
            double *Rbuff = &(*R_iajbkc[ijk])(0,0);
            double *Dbuff = &(*denom_ijk_[ijk])(0,0);
            for (int vp = 0; vp < ntno_ijk * ntno_ijk * ntno_ijk; ++vp) {
                (*Tbuff) -= (*Rbuff) / (*Dbuff);
                ++Rbuff, ++Tbuff, ++Dbuff;
            }

            // account for symmetry
            t_symmetrizer(T_iajbkc_, ijk);
        }

        // => DIIS Extrapolation <= //
        auto T_iajbkc_flat = flatten_mats(T_iajbkc_);
        auto R_iajbkc_flat = flatten_mats(R_iajbkc);

        if (iteration == 1) {
            diis.set_error_vector_size(R_iajbkc_flat.get());
            diis.set_vector_size(T_iajbkc_flat.get());
        }

        diis.add_entry(R_iajbkc_flat.get(), T_iajbkc_flat.get());
        diis.extrapolate(T_iajbkc_flat.get());

        copy_flat_mats(T_iajbkc_flat, T_iajbkc_);

        // evaluate convergence
        e_prev = e_curr;
        // Compute LCCSD(T) energy
        e_curr = e_lccsd_ + compute_t_energy();

        double r_curr = *max_element(R_iajbkc_rms.begin(), R_iajbkc_rms.end());

        r_converged = fabs(r_curr) < options_.get_double("R_CONVERGENCE");
        e_converged = fabs(e_curr - e_prev) < options_.get_double("E_CONVERGENCE");

        outfile->Printf("  @LCCSD(T) iter %3d: %16.12f %10.3e %10.3e\n", iteration, e_curr, e_curr - e_prev, r_curr);

        iteration++;

        if (iteration > max_iteration) {
            throw PSIEXCEPTION("Maximum DLPNO iterations exceeded.");
        }
    }

    e_lccsd_t_ = e_curr;

    timer_off("LCCSD(T) Iterations");
}

double DLPNOCCSD_T::compute_energy() {
    timer_on("DLPNO-CCSD(T)");
    // Run DLPNO-CCSD
    double e_dlpno_ccsd = DLPNOCCSD::compute_energy();

    tno_transform();
    compute_tno_overlaps();
    compute_W_iajbkc();
    compute_lccsd_t0();
    if (!options_.get_bool("T0_APPROXIMATION")) lccsd_t_iterations();

    double e_scf = reference_wavefunction_->energy();
    double e_ccsd_t_corr = e_lccsd_t_ + de_lmp2_ + de_dipole_ + de_pno_total_;
    double e_ccsd_t_total = e_scf + e_ccsd_t_corr;

    set_scalar_variable("CCSD(T) CORRELATION ENERGY", e_ccsd_t_corr);
    set_scalar_variable("CURRENT CORRELATION ENERGY", e_ccsd_t_corr);
    set_scalar_variable("CCSD(T) TOTAL ENERGY", e_ccsd_t_total);
    set_scalar_variable("CURRENT ENERGY", e_ccsd_t_total);

    print_results();

    timer_off("DLPNO-CCSD(T)");

    return e_ccsd_t_total;
}

void DLPNOCCSD_T::print_results() {
    outfile->Printf("  \n");
    outfile->Printf("  Total DLPNO-CCSD(T) Correlation Energy: %16.12f \n", e_lccsd_t_ +  de_lmp2_ + de_pno_total_ + de_dipole_);
    outfile->Printf("    DLPNO-CCSD Contribution:              %16.12f \n", e_lccsd_);
    outfile->Printf("    DLPNO-(T) Contribution:               %16.12f \n", e_lccsd_t_ - e_lccsd_);
    outfile->Printf("    LMP2 Weak Pair Correction:            %16.12f \n", de_lmp2_);
    outfile->Printf("    LMO Truncation Correction:            %16.12f \n", de_dipole_);
    outfile->Printf("    PNO Truncation Correction:            %16.12f \n", de_pno_total_);
    outfile->Printf("    Andy Jiang... FOR THREEEEEEEEEEE!!!\n\n\n");
    outfile->Printf("  @Total DLPNO-CCSD(T) Energy: %16.12f \n", variables_["SCF TOTAL ENERGY"] +
                                                                    de_lmp2_ + e_lccsd_t_ + de_pno_total_ + de_dipole_);
}

}
}
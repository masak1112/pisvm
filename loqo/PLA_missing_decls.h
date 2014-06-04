// D. Brugger, september 2006
// PLA_missing_decls.h - Contains missing declarations for some functions
// in PLAPACKR30.
// $Id: PLA_missing_decls.h 573 2010-12-29 10:54:20Z dome $
//
// Copyright (C) 2006 Dominik Brugger
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

#ifndef __PLA__MISSING_H
#define __PLA__MISSING_H

int PLA_Syrk(int uplo, int trans, PLA_Obj alpha, PLA_Obj a, PLA_Obj beta,
	     PLA_Obj c);
int PLA_Trsm(int side, int uplo, int transa, int diag, PLA_Obj alpha,
	     PLA_Obj a, PLA_Obj b);
int PLA_Trsv(int uplo, int trans, int diag, PLA_Obj a, PLA_Obj x);
int PLA_Trmv(int uplo, int trans, int diag, PLA_Obj a, PLA_Obj x);
int PLA_Copy(PLA_Obj from, PLA_Obj to);
int PLA_Chol(int uplo, PLA_Obj a);
int PLA_Gemv(int trans, PLA_Obj alpha, PLA_Obj a, PLA_Obj x, PLA_Obj beta,
	     PLA_Obj y);
int PLA_Temp_comm_all_info(PLA_Template templ, MPI_Comm *comm,
			   int *rank, int *numnodes);
int PLA_Temp_comm_all(PLA_Template templ, MPI_Comm *comm);
int PLA_Temp_create(int nb_distr, int zero_or_one, PLA_Template *templ);
int PLA_Dot(PLA_Obj x, PLA_Obj y, PLA_Obj alpha);
int PLA_Axpy(PLA_Obj alpha, PLA_Obj x, PLA_Obj y);
int PLA_Iamax(PLA_Obj x, PLA_Obj xmax, PLA_Obj k);

int PLA_Comm_1D_to_2D(MPI_Comm co, int nprows, int npcols, MPI_Comm *comm);
int PLA_Init(MPI_Comm comm);
int PLA_Finalize();

int PLA_Set_error_checking(int error, int parameters, int sequential, 
			   int r12 );
int PLA_Get_error_checking(int *error, int *parameters, int *sequential, 
			   int *r12 );

#endif

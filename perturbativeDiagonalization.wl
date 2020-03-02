(* ::Package:: *)

(* ::Title:: *)
(*Dynamical perturbation theory for eigenvalue problems*)
(*arXiv:2002.12872*)


(* ::Text:: *)
(*Anton Zadorin, Maxim Kenmoe and Matteo Smerlak*)
(*Max Planck Institute for Mathematics in the Sciences *)
(**)
(*Mar. 2, 2020*)


(* ::Section:: *)
(*Helper functions*)


(* ::Text:: *)
(*Zero-out the off-diagonal elements of a matrix and return a sparse array:*)


(* ::Input::Initialization:: *)
diag[B_]:=SparseArray[{i_,i_}:>Normal[Diagonal[B]][[i]],Dimensions@B]
offDiag[B_]:=B-diag@B


(* ::Section:: *)
(*Perturbative diagonalization*)


(* ::Text:: *)
(*Compute the matrix \[Theta] of inverse spectral gaps for the unperturbed matrix:*)


(* ::Input::Initialization:: *)
inverseGaps[d_]:=Block[{gaps},
gaps=Outer[Plus,d,-d]; Do[gaps[[i,i]]=\[Infinity],{i,Length@d}];1/gaps]

inverseGapsCompiled=Compile[{{x,_Real,1}},Outer[If[#1==#2,0,1/(#1-#2)]&,x,x],CompilationTarget->"C",Parallelization->True];

inverseGapsOneLineCompiled=Compile[{{\[Epsilon],_Real},{x,_Real,1}},If[#==\[Epsilon],0,1/(\[Epsilon]-#)]&/@x,CompilationTarget->"C",Parallelization->True];


(* ::Subsection:: *)
(*Rayleigh-Schr\[ODoubleDot]dinger series*)


(* ::Input::Initialization:: *)
rayleighSchrodingerDiagonalization[M_,D_:Nothing,normalize_:False,terminationThreshold_:100,maxIterations_:500]:=Catch@Block[{id,A,\[CapitalDelta],N,M0,\[Gamma],\[Theta],k,errors,error,G,$MinPrecision},

If[Precision@M<\[Infinity],$MinPrecision=Precision@M];
M0=If[D==Nothing,diag@M,D];

N=First@Dimensions@M;
\[CapitalDelta]=Transpose[M-M0];
id=IdentityMatrix[N,Head@M];

\[Theta]=If[Head@M===SparseArray,SparseArray,Identity]@If[Precision@M<=MachinePrecision,inverseGapsCompiled,inverseGaps]@Normal@Diagonal@M0;


\[Gamma][k_]:=\[Gamma][k]={#,#.\[CapitalDelta],\[Theta]*#}&[
(\[Theta]*\[Gamma][k-1][[2]]-Sum[diag[\[Gamma][s][[2]]].\[Gamma][k-1-s][[3]],{s,0,k-1}])
];
\[Gamma][0]={id,\[CapitalDelta],\[Theta]*id};


k=0;
{G,errors}=Reap[NestWhile[(k++;#+\[Gamma][k])&,\[Gamma][0],(error=Norm[First[#1-#2],"Frobenius"]/Norm[First[#1],"Frobenius"];
If[Max@Abs[First@#1]>terminationThreshold,Throw["Error: an entry exceeded "<>ToString[terminationThreshold]],Sow[error]];
error>2*10^(-Precision@M))&,2,maxIterations]];


If[Last@errors>2*10^(-Precision@M),Throw["Warning: still no convergence after "<>ToString[maxIterations]<>" iterations"]];


A=G[[1]];
Association[
"eigenvalues"->Normal@Diagonal[M0+A.\[CapitalDelta]],
"eigenvectors"->Normal@If[normalize,Normalize/@A,A],
"errors"->errors[[1]]
]

]


(* ::Subsection:: *)
(*Dynamical perturbation theory*)


(* ::Text:: *)
(*Complete set of eigenvectors:*)


(* ::Input::Initialization:: *)
dynamicalDiagonalization[M_,D_:Nothing,normalize_:False,terminationThreshold_:100,maxIterations_:500]:=Catch@Block[{A,id,\[CapitalDelta],N,F,\[Theta],M0,errors,error,$MinPrecision},

If[Precision@M<\[Infinity],$MinPrecision=Precision@M];
M0=If[D==Nothing,diag@M,D];

N=First@Dimensions@M;
\[CapitalDelta]=Transpose[M-M0];
id=IdentityMatrix[N,Head@M];

\[Theta]=If[Head@M===SparseArray,SparseArray,Identity]@If[Precision@M<=MachinePrecision,inverseGapsCompiled,inverseGaps]@Normal@Diagonal@M0;

F[A_]:=id+\[Theta]*(#-diag[#].A)&[A.\[CapitalDelta]];

{A,errors}=Reap@FixedPoint[F,id,maxIterations,SameTest->((error=Norm[#1-#2,"Frobenius"]/Norm[#1,"Frobenius"];If[Max@Abs[#1]>terminationThreshold,Throw["Error: an entry exceeded "<>ToString[terminationThreshold]],Sow[error]];error<2*10^(-Precision@M))&)];


If[Last@errors[[1]]>2*10^(-Precision@M),Throw["Warning: still no convergence after "<>ToString[maxIterations]<>" iterations"]];


Association[
"eigenvalues"->Normal@Diagonal[M0+A.\[CapitalDelta]],
"eigenvectors"->Normal@If[normalize,Normalize/@A,A],
"errors"->errors[[1]]
]

]


(* ::Text:: *)
(*A single eigenvector:*)


(* ::Input::Initialization:: *)
dynamicalEigenvector[M_,n_,D_:Nothing,normalize_:False,terminationThreshold_:100,maxIterations_:500]:=Catch@Block[{a,id,\[CapitalDelta],M0,N,f,\[Theta],errors,error,$MinPrecision},

If[Precision@M<\[Infinity],$MinPrecision=Precision@M];
M0=If[D==Nothing,diag@M,D];

N=First@Dimensions@M;
\[CapitalDelta]=M-M0;
id=IdentityMatrix[N,Head@M];

\[Theta]=If[Head@M===SparseArray,SparseArray,Identity]@If[Precision@M<=MachinePrecision,inverseGapsOneLineCompiled[M[[n,n]],#]&]@Normal@Diagonal@M0;

f[b_]:=id[[n,All]]+\[Theta]*(#-#[[n]]b)&[\[CapitalDelta].b];


{a,errors}=Reap@FixedPoint[f,id[[n]],maxIterations,SameTest->((error=Norm[#1-#2]/Norm[#1];If[Max@Abs[#1]>terminationThreshold,Throw["Error: an entry exceeded "<>ToString[terminationThreshold]],Sow[error]];error<2*10^(-Precision@M))&)];


If[Last@errors[[1]]>2*10^(-Precision@M),Throw["Warning: still no convergence after "<>ToString[maxIterations]<>" iterations"]];


Association[
"eigenvalue"->M0[[n,n]]+(\[CapitalDelta].a)[[n]],
"eigenvector"->Normal@If[normalize,Normalize@a,a],
"errors"->errors[[1]]
]

]

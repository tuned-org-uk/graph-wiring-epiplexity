(* ::Package:: *)

(* :Title: ArrowSpaceEpiplexity *)
(* :Context: ArrowSpaceEpiplexity` *)
(* :Author:  Lorenzo @Mec-iS *)
(* :Summary: Epiplexity-style structural information tools for ArrowSpace-style feature-space Laplacians. *)
(* :Package Version: 0.1 *)
(* :Mathematica Version: 13.3 *)
(* :History: 2026-03-07 initial version. *)
(* :Discussion:
   This package implements:
   - Prefix-free description length for the ArrowSpace model |P_AS|
   - LGMRF probabilistic wrapper P_AS(x) on the feature Laplacian L_F
   - Two-part MDL decomposition over datasets
   - Diagnostics distinguishing structural information vs metadata
   - Multi-class engine (search, classification, anomaly detection, etc.)
   - LLM-scale data engineering utilities (data selection, drift, OOD)
*)

BeginPackage["ArrowSpaceEpiplexity`"]

(* ==================== USAGE MESSAGES ==================== *)

ArrowSpaceEliasGammaBits::usage =
"ArrowSpaceEliasGammaBits[x] gives the Elias gamma code length in bits \
for a positive integer x, used for prefix-free model encoding.";

ArrowSpaceDescriptionLength::usage =
"ArrowSpaceDescriptionLength[N, F, C0, k, b, Feff] returns an association \
with the structural description length |P_AS| of the ArrowSpace model and \
related components (header bits, centroid bits, Laplacian topology bits).";

ArrowSpaceProbabilisticModel::usage =
"ArrowSpaceProbabilisticModel[L_F, beta, gamma] builds a Laplacian-constrained \
Gaussian Markov Random Field (LGMRF) model on the feature-space Laplacian L_F. \
The result is an association with keys:
  \"EvaluateLogProb\", \"DirichletEnergy\", \"RayleighQuotient\", \
  \"TimeBoundedEntropy\", \"Sample\", \"DescriptionLengthBits\", etc.";

ArrowSpaceBuildFeatureLaplacian::usage =
"ArrowSpaceBuildFeatureLaplacian[X, k, sigma] builds the feature-space \
Laplacian L_F (F x F) from an item-feature matrix X (N x F) via k-NN on \
transposed features with Gaussian kernel weights.";

ArrowSpaceMDLToolkit::usage =
"ArrowSpaceMDLToolkit[X, k, beta, gamma, sigma, nCentroids] builds a toolkit \
association for dataset X, with keys:
  \"L_F\", \"Model\", \"Rayleigh\", \"Entropy\", \
  \"StructuralBits\", \"TotalEntropyBits\", \"MDLTotalBits\", \
  \"RawBits\", \"CompressionRatio\", \"SpectralGap\".";

ArrowSpaceCompressionTest::usage =
"ArrowSpaceCompressionTest[toolkit] returns an association summarising the \
two-part MDL decomposition (in KB), compression ratio, spectral gap, and \
whether MDL_T < raw bits (structural vs metadata test).";

ArrowSpaceStructuralDiagnostics::usage =
"ArrowSpaceStructuralDiagnostics[toolkit, name] runs three tests \
(compression ratio, spectral gap, Rayleigh CV) and prints a verdict on \
whether L_F is structural information or metadata.";

ArrowSpaceMultiClassEngine::usage =
"ArrowSpaceMultiClassEngine[X, k, sigma, beta, gamma] builds a multi-class \
engine association that reuses a single L_F for:
  \"Search\", \"LabelPropagation\", \"AnomalyScores\", \
  \"Diffuse\", \"LaplacianEigenmaps\", \"DataValuation\".";

ArrowSpaceLLMToolset::usage =
"ArrowSpaceLLMToolset[engine] builds a production-focused toolset association \
for LLM-scale data engineering, with functions:
  \"SelectForEpiplexity\", \"SpectralAnomalyGuard\", \
  \"SpectralFingerprint\", \"DetectDrift\".";

ArrowSpaceObserverSweep::usage =
"ArrowSpaceObserverSweep[X, kValues] sweeps k over a list of graph densities, \
returning a table of |P_AS| (KB), mean H_T, compression ratio, and spectral gap.";

Begin["`Private`"]

(* ==================== BASIC ENCODING ==================== *)

ArrowSpaceEliasGammaBits[x_Integer?Positive] :=
  2*Floor[Log2[Max[1, x]]] + 1;

ArrowSpaceDescriptionLength[
   N_Integer, F_Integer,
   C0_: Automatic, k_: 16,
   b_: 32, Feff_: Automatic] :=
 Module[{centroids, Fe, headerBits, paramBits, centroidBits, topoBits, total, raw},
  centroids = If[C0 === Automatic, Max[100, Min[2000, Floor[2*Sqrt[N]]]], C0];
  Fe = If[Feff === Automatic, F, Feff];
  headerBits = Total[ArrowSpaceEliasGammaBits /@ {N, F, centroids, k}];
  paramBits = 64 + 8 + 32;
  centroidBits = centroids*Fe*b;
  topoBits = F*k*(Ceiling[Log2[Max[2, F]]] + b);
  total = headerBits + paramBits + centroidBits + topoBits;
  raw = N*F*b;
  <|
    "N" -> N, "F" -> F, "C0" -> centroids, "k" -> k,
    "HeaderBits" -> headerBits,
    "CentroidBits" -> centroidBits,
    "TopologyBits" -> topoBits,
    "TotalBits" -> total,
    "TotalKB" -> N[total/8/1024],
    "RawKB" -> N[raw/8/1024],
    "CompressionRatio" -> N[raw/total]
  |>
 ]

(* ==================== PROBABILISTIC MODEL ==================== *)

ArrowSpaceProbabilisticModel[LapF_?MatrixQ, beta_: 1., gamma_: 1.*^-3] :=
 Module[{F, L, Q, eigs, logDetQ, logZ},
  F = Length[LapF];
  L = LapF;
  Q = beta*L + gamma*IdentityMatrix[F];

  eigs = Eigenvalues[N[Q]];
  logDetQ = Total[Log[Abs[eigs]]];
  logZ = 1/2*F*Log[2*Pi] - 1/2*logDetQ;

  <|
    "F" -> F,
    "Beta" -> beta,
    "Gamma" -> gamma,
    "L" -> L,
    "Q" -> Q,
    "LogDetQ" -> logDetQ,
    "LogZ" -> logZ,
    "EvaluateLogProb" ->
      Function[{x},
        Module[{qx = x . (Q . x)},
          -1/2*qx - logZ
        ]
      ],
    "DirichletEnergy" ->
      Function[{x}, x . (L . x)],
    "RayleighQuotient" ->
      Function[{x},
        Module[{num = x . (L . x), den = x . x},
          If[den <= 10^-12, 0., num/den]
        ]
      ],
    "TimeBoundedEntropy" ->
      Function[{x},
        -("EvaluateLogProb"[x]/Log[2])
      ],
    "Sample" ->
      Function[{nSamples: 1},
        Module[{cov, z},
          cov = Inverse[Q];
          z = RandomVariate[
            MultinormalDistribution[ConstantArray[0, F], cov],
            nSamples
          ];
          If[nSamples == 1, First[z], z]
        ]
      ],
    "DescriptionLengthBits" ->
      Function[{C0, k, b: 32},
        Module[{header, centroid, topo, params},
          header = Total[ArrowSpaceEliasGammaBits /@ {F, C0, k}];
          centroid = C0*F*b;
          topo = F*k*(Ceiling[Log2[Max[2, F]]] + b);
          params = 64 + 8 + 32;
          header + centroid + topo + params
        ]
      ]
  |>
 ]

(* ==================== FEATURE LAPLACIAN ==================== *)

ArrowSpaceBuildFeatureLaplacian[X_?MatrixQ, k_: 5, sigma_: 1.] :=
 Module[{Xt, Fnodes, distMat, W, D, L},
  Xt = Transpose[X];
  Fnodes = Length[Xt];

  distMat = DistanceMatrix[Xt, Xt, DistanceFunction -> EuclideanDistance];
  Do[distMat[[i, i]] = Infinity, {i, 1, Fnodes}];

  W = ConstantArray[0., {Fnodes, Fnodes}];
  Do[
   Module[{row = distMat[[i]], nbrs},
    nbrs = Ordering[row, k];
    Do[
      Module[{j = n, w},
        w = Exp[-row[[j]]^2/(2*sigma^2)];
        W[[i, j]] = w; W[[j, i]] = w;
      ],
      {n, nbrs}
    ];
   ],
   {i, 1, Fnodes}
  ];

  D = DiagonalMatrix[Total[W, {2}]];
  L = D - W;
  SparseArray[L]
 ]

(* ==================== MDL TOOLKIT ==================== *)

ArrowSpaceMDLToolkit[
   X_?MatrixQ, k_: 5,
   beta_: 1., gamma_: 1.*^-3,
   sigma_: 1., nCentroids_: Automatic] :=
 Module[{N, F, C0, LF, mdl, rq, ent, logp},
  {N, F} = Dimensions[X];
  C0 = If[nCentroids === Automatic, Max[10, Min[200, Floor[2*Sqrt[N]]]], nCentroids];
  LF = ArrowSpaceBuildFeatureLaplacian[X, k, sigma];
  mdl = ArrowSpaceProbabilisticModel[Normal[LF], beta, gamma];

  rq = mdl["RayleighQuotient"] /@ X;
  ent = mdl["TimeBoundedEntropy"] /@ X;
  logp = mdl["EvaluateLogProb"] /@ X;

  <|
    "X" -> X,
    "N" -> N,
    "F" -> F,
    "k" -> k,
    "C0" -> C0,
    "L_F" -> LF,
    "Model" -> mdl,
    "Rayleigh" -> rq,
    "Entropy" -> ent,
    "LogProbs" -> logp,
    "StructuralBits" -> mdl["DescriptionLengthBits"][C0, k],
    "TotalEntropyBits" -> Total[ent],
    "RawBits" -> N*F*32,
    "MDLTotalBits" :> (mdl["DescriptionLengthBits"][C0, k] + Total[ent]),
    "CompressionRatio" :> (N*F*32/(mdl["DescriptionLengthBits"][C0, k] + Total[ent])),
    "SpectralGap" :>
      Module[{evals},
        evals = Sort[Abs@Eigenvalues[Normal[LF], 6]];
        If[Length[evals] > 1, evals[[2]], 0.]
      ]
  |>
 ]

ArrowSpaceCompressionTest[toolkit_Association] :=
 Module[{mdl, raw, gap},
  mdl = toolkit["MDLTotalBits"];
  raw = toolkit["RawBits"];
  gap = toolkit["SpectralGap"];
  <|
    "StructuralBitsKB" -> toolkit["StructuralBits"]/8/1024,
    "EntropyBitsKB" -> toolkit["TotalEntropyBits"]/8/1024,
    "MDLTotalKB" -> mdl/8/1024,
    "RawBitsKB" -> raw/8/1024,
    "CompressionRatio" -> raw/mdl,
    "SpectralGap" -> gap,
    "PassesCompression" -> (mdl < raw)
  |>
 ]

ArrowSpaceStructuralDiagnostics[toolkit_Association, name_String:"Dataset"] :=
 Module[{rq, cv, r},
  rq = toolkit["Rayleigh"];
  cv = StandardDeviation[rq]/(Mean[rq] + 10^-12);
  r = ArrowSpaceCompressionTest[toolkit];

  Print[StringRepeat["=", 60]];
  Print["Diagnostics: ", name];
  Print[StringRepeat["=", 60]];

  Print[If[r["PassesCompression"], "\[Checkmark] ", "✗ "],
    "Compression Ratio = ", NumberForm[r["CompressionRatio"], {4, 2}],
    " | MDL ", NumberForm[r["MDLTotalKB"], {6, 2}],
    " KB vs Raw ", NumberForm[r["RawBitsKB"], {6, 2}], " KB"
  ];

  Print[If[r["SpectralGap"] > 10^-3, "\[Checkmark] ", "✗ "],
    "Spectral Gap \[Lambda]2 = ", NumberForm[r["SpectralGap"], {5, 4}],
    "  (threshold 1e-3)"
  ];

  Print[If[cv > 0.05, "\[Checkmark] ", "✗ "],
    "Rayleigh CV = ", NumberForm[cv, {4, 3}],
    " | mean \[Lambda] = ", NumberForm[Mean[rq], {5, 4}],
    " | std = ", NumberForm[StandardDeviation[rq], {5, 4}]
  ];

  Print["----------------------------------------------"];
  If[r["PassesCompression"] && r["SpectralGap"] > 10^-3 && cv > 0.05,
    Print["VERDICT: L_F IS STRUCTURAL INFORMATION"],
    Print["VERDICT: L_F IS METADATA (some tests failed)"]
  ];
 ]

(* ==================== MULTI-CLASS ENGINE ==================== *)

ArrowSpaceMultiClassEngine[
   X_?MatrixQ, k_: 5, sigma_: 1., beta_: 1., gamma_: 1.*^-3] :=
 Module[{N, F, LF, mdl, lambdas},
  {N, F} = Dimensions[X];
  LF = ArrowSpaceBuildFeatureLaplacian[X, k, sigma];
  mdl = ArrowSpaceProbabilisticModel[Normal[LF], beta, gamma];
  lambdas = mdl["RayleighQuotient"] /@ X;

  <|
    "X" -> X, "N" -> N, "F" -> F,
    "L_F" -> LF,
    "Model" -> mdl,
    "Lambdas" -> lambdas,

    "Search" ->
      Function[{q, topK: 5, alpha: 0.6},
        Module[{qNorm, xNorm, cosine, qLam, lamDist, lamSim, combined, idx},
          qNorm = q/Norm[q + 10.^-12];
          xNorm = X/Norm[#, 2] & /@ X;
          cosine = xNorm.qNorm;
          qLam = mdl["RayleighQuotient"][q];
          lamDist = Abs[lambdas - qLam];
          lamSim = 1/(1 + lamDist);
          combined = alpha*cosine + (1 - alpha)*lamSim;
          idx = Ordering[combined, -topK];
          Transpose[{idx, combined[[idx]]}]
        ]
      ],

    "LabelPropagation" ->
      Function[{labelAssoc, mu: 0.1, nIter: 20},
        Module[{f, labeled},
          f = ConstantArray[0., N];
          Do[f[i + 1] = labelAssoc[i], {i, Keys[labelAssoc]}];
          labeled = Keys[labelAssoc] + 1;
          Do[
            Do[
              If[MemberQ[labeled, i], Continue[]];
              Module[{d = Abs[lambdas - lambdas[[i]]], nbrs},
                nbrs = Ordering[d, 5][[2 ;; 5]];
                f[[i]] = (1 - mu)*f[[i]] + mu*Mean[f[[nbrs]]];
              ],
              {i, 1, N}
            ],
            {nIter}
          ];
          Sign[f]
        ]
      ],

    "AnomalyScores" ->
      Function[{pct: 95},
        Module[{thr},
          thr = Quantile[lambdas, pct/100.];
          Boole[lambdas > thr]
        ]
      ],

    "Diffuse" ->
      Function[{signal, eta: 0.05, steps: 10},
        Module[{x = signal, L = LF},
          Do[x = x - eta.(L.x), {steps}];
          x
        ]
      ],

    "LaplacianEigenmaps" ->
      Function[{d: 2},
        Module[{keigs, evals, evecs, order, nontrivial},
          keigs = Min[d + 2, F - 1];
          {evals, evecs} = Eigensystem[Normal[LF], keigs, Method -> "Arnoldi"];
          order = Ordering[evals];
          evecs = evecs[[order]];
          nontrivial = Transpose[evecs[[2 ;; d + 1]]];
          X.nontrivial
        ]
      ],

    "DataValuation" ->
      Function[{},
        mdl["TimeBoundedEntropy"] /@ X
      ]
  |>
 ]

(* ==================== LLM TOOLSET ==================== *)

ArrowSpaceLLMToolset[engine_Association] :=
 Module[{X = engine["X"], lambdas = engine["Lambdas"], mdl = engine["Model"]},
  <|
    "SelectForEpiplexity" ->
      Function[{budgetFrac: 0.5},
        Module[{ht, lo, hi, idx, nBudget},
          ht = mdl["TimeBoundedEntropy"] /@ X;
          {lo, hi} = Quantile[ht, {0.25, 0.75}];
          idx = Flatten@Position[ht, _?(lo <= # <= hi &)];
          nBudget = Floor[Length[X]*budgetFrac];
          If[Length[idx] > nBudget,
            idx[[Round@Rescale[Range[nBudget], {1, nBudget}, {1, Length[idx]}]]],
            idx
          ]
        ]
      ],

    "SpectralAnomalyGuard" ->
      Function[{q, thresholdSigma: 2.},
        Module[{qLam, mu, sd, z},
          qLam = mdl["RayleighQuotient"][q];
          mu = Mean[lambdas]; sd = StandardDeviation[lambdas];
          z = (qLam - mu)/(sd + 10^-12);
          <|
            "QueryLambda" -> qLam,
            "PopulationMu" -> mu,
            "PopulationSD" -> sd,
            "ZScore" -> z,
            "IsOOD" -> Abs[z] > thresholdSigma
          |>
        ]
      ],

    "SpectralFingerprint" ->
      Function[{nEigs: 6},
        Module[{ke, evals, lamQ},
          ke = Min[nEigs, engine["F"] - 1];
          evals = Sort[Abs@Eigenvalues[Normal[engine["L_F"]], ke]];
          lamQ = Quantile[lambdas, {0.1, 0.25, 0.5, 0.75, 0.9}];
          <|
            "LaplacianSpectrum" -> evals,
            "SpectralGap" -> If[Length[evals] > 1, evals[[2]], 0.],
            "LambdaQuantiles" -> lamQ,
            "LambdaMean" -> Mean[lambdas],
            "LambdaStd" -> StandardDeviation[lambdas]
          |>
        ]
      ],

    "DetectDrift" ->
      Function[{fp0, fp1, gapThreshold: 0.1},
        Module[{gapChange, meanShift, drift},
          gapChange = Abs[fp1["SpectralGap"] - fp0["SpectralGap"]];
          meanShift = Abs[fp1["LambdaMean"] - fp0["LambdaMean"]];
          drift = Or[gapChange > gapThreshold, meanShift > 0.5];
          <|
            "SpectralGapChange" -> gapChange,
            "LambdaMeanShift" -> meanShift,
            "DriftDetected" -> drift
          |>
        ]
      ]
  |>
 ]

(* ==================== OBSERVER SWEEP ==================== *)

ArrowSpaceObserverSweep[X_?MatrixQ, kValues_List] :=
  Table[
    Module[{tk = ArrowSpaceMDLToolkit[X, k]},
      <|
        "k" -> k,
        "StructuralKB" -> tk["StructuralBits"]/8/1024,
        "EntropyMean" -> Mean[tk["Entropy"]],
        "Compression" -> tk["CompressionRatio"],
        "SpectralGap" -> tk["SpectralGap"]
      |>
    ],
    {k, kValues}
  ]

End[]; (* `Private` *)

EndPackage[];

(* ::Package:: *)

(* : Title : BuffonNeedle*)
(* : Author : Enis Siniksaran*)
(* : Mathematica Version : 5.2*)
(* : Modified for compatibility with M5 or M6 *)
(* : Date : January, 2006*)
(* : Summary : This package is designed to carry out the Buffon's Needle experiments 
for single, double and triple grids.*)

If[$VersionNumber<6.0,
	BeginPackage["BuffonNeedle`","Statistics`ContinuousDistributions`"],
	{BeginPackage["BuffonNeedle`"]}
];

SingleGrid::usage = "SingleGrid[n,r] implements the single-grid Buffon's experiment by Monte Carlo Simulation.
It gives a table showing the number and the frequency ratios of two possible outcomes together with the theoretical 
probabilities and the estimate of Pi. The function also gives the picture of the needles thrown onto the single grid. 
Here, n is the number of the needles and r is the ratio of the needle length to the grid height(i.e. r = l / d). 
n can be any integer while r is a real number on the interval (0,1]. ";

DoubleGrid::usage = "DoubleGrid[n,r] implements the double-grid Buffon's experiment by Monte Carlo Simulation.
It gives a table showing the number and the frequency ratios of three possible outcomes together with the theoretical 
probabilities and the estimate of Pi. The function also gives the picture of the needles thrown onto the double grid. 
Here, n is the number of the needles and r is the ratio of the needle length to the grid height (i.e. r = l / d). 
n can be any integer while r is a real number on the interval (0,1]. ";

TripleGrid::usage = "TripleGrid[n,r] implements the triple-grid Buffon's experiment by Monte Carlo Simulation.
It gives a table showing the number and the frequency ratios of four possible outcomes together with the theoretical 
probabilities and the estimate of Pi. The function also gives the picture of the needles thrown onto the triple grid. 
Here, n is the number of the needles and r is the ratio of the needle length to the grid height (i.e. r = l / d). 
n can be any integer while r is a real number on the interval (0,1]. ";

Begin["`Private`"]
Off[Power::"infy"]
Off[General::"spell1"]
rmb::"urp" = "r must be a number between zero and one ";
int::"fls" = "n must be a positive integer number ";

If[$VersionNumber<6.0,
	Unprotect[UniformDistribution];
	UniformDistribution[{min_,max_}]=UniformDistribution[min,max];
	Protect[UniformDistribution];
	RandomReal[{min_,max_}]:=Random[Real,{min,max}];
	RandomReal[{min_,max_},n_]:=Table[Random[Real,{min,max}],{n}];
	RandomReal[dist_]:=Random[dist];
	RandomReal[dist_,n_]:=RandomArray[dist,n]];

SingleGrid[n_, r_] := 
    Module[{l, d, data, lines, res, hit0, hit1, s, frame, probs, wp, esti},
      l = r*d; d = 1;
      If[IntegerQ[n] == False || Positive[n] == False, Message[int::"fls"]];
      If[r > 1 || r <= 0, Message[rmb::"urp"]];
      If[r > 1 || r <= 0 || IntegerQ[n] == False || Positive[n] == False, 
        Abort[]];
      data = 
        Transpose[{Table[{ RandomReal[{1, 4}], RandomReal[{1, 3}]}, {n}], 
            RandomReal[{0, Pi}, n]}];
lines = (Function[s, 
                  First[#] + 
                    s r/2 Through[{Cos, Sin}[Last[#]]]] /@ {1, -1}) & /@data;
res = Table[
          If[Ceiling[lines][[i, 1, 2]] == Ceiling[lines][[i, 2, 2]], 1, 
            0], {i, 1, n}];
hit0 = Flatten[Position[res, 1]];
      hit1 = Complement[Range[n], hit0];
frame = {{{4.5, 1}, {.5, 1}}, {{.5, 1}, {4.5, 1}}, {{.5, 2}, {4.5, 2}}, {{.5, 
              3}, {4.5, 3}}};
      probs = {1 - 2*r*t, 2*r*t};
      esti = N[(2*r)/(1 - (Length[hit0]/n)),15];
      wp = 1/(55 + Log[n^4]);
      esti
      ];
      
DoubleGrid[n_, r_] := 
    Module[{l, d, data, lines, res, hit0, hit1, hit2, s, main, ext, probs, 
        wp},
      l = r*d; d = 1;
      If[IntegerQ[n] == False || Positive[n] == False, Message[int::"fls"]];
      If[r > 1 || r <= 0, Message[rmb::"urp"]];
      If[r > 1 || r <= 0 || IntegerQ[n] == False || Positive[n] == False, 
        Abort[]];
data = Transpose[{Table[{ RandomReal[{1, 4}], RandomReal[{1, 3}]}, {n}], 
            RandomReal[{0, Pi}, n]}];
lines = (Function[s, 
                  First[#] + 
                    s r/2 Through[{Cos, Sin}[Last[#]]]] /@ {1, -1}) & /@data;
resabs = Table[
          If[Ceiling[lines][[i, 1, 1]] == Ceiling[lines][[i, 2, 1]], 1, 
            0], {i, 1, n}];
resord = Table[
          If[Ceiling[lines][[i, 1, 2]] == Ceiling[lines][[i, 2, 2]], 1, 
            0], {i, 1, n}];
hitord = Flatten[Position[resord, 1]];
hitabs = Flatten[Position[resabs, 1]];
hit0 = Intersection[hitord, hitabs];
hit1 = Complement[Union[hitord, hitabs], Intersection[hitord, hitabs]];
hit2 = Complement[Range[n], Union[hitord, hitabs]];
main = {{{1, 1}, {4, 1}}, {{1, 2}, {4, 2}}, {{1, 3}, {4, 3}}, {{1, 1}, {1, 
              3}}, {{2, 1}, {2, 3}}, {{3, 1}, {3, 3}}, {{4, 1}, {4, 3}}};
ext = {{{0.5, 1}, {1, 1}}, {{0.5, 2}, {1, 2}}, {{0.5, 3}, {1, 3}}, {{1, 
              3}, {1, 3.5}}, {{1, 1}, {1, 0.5}}, {{2, 3}, {2, 3.5}}, {{2, 
              1}, {2, 0.5}}, {{3, 1}, {3, 0.5}}, {{4, 3}, {4.5, 3}}, {{4, 
              2}, {4.5, 2}}, {{4, 1}, {4.5, 1}}, {{4, 3}, {4, 3.5}}, {{4, 
              1}, {4, 0.5}}, {{3, 3}, {3, 3.5}}};
probs = {1 - r*(4 - r)*t, 2*r*(2 - r)*t, (r^2)*t};
esti = N[(4*r - r^2)/(1 - Length[hit0]/n), 15];
      wp = 1/(55 + Log[n^4]);
      esti
      ];
      
TripleGrid[n_, r_] := 
    Module[{l, d, main, ext, frame, absords, data, lines, res, hit0, hit1, s, 
        probs, wp},
      d = 1; l = r*d;
      rmb::"urp" = "r must be a number between zero and one ";
      int::"fls" = "n must be a positive integer number ";
      If[IntegerQ[n] == False || Positive[n] == False, Message[int::"fls"]];
      If[r > 1 || r <= 0, Message[rmb::"urp"]];
      If[r > 1 || r <= 0 || IntegerQ[n] == False || Positive[n] == False, 
        Abort[]];     
      main = {{{1.345, 2}, {1.923, 3}}, {{1.923, 1}, {1.345, 2}}, {{1.923, 
              3}, {3.077, 3}}, {{2.5, 2}, {1.345, 2}}, {{2.5, 2}, {1.922, 
              1}}, {{2.5, 2}, {1.923, 3}}, {{2.5, 2}, {3.077, 1}}, {{2.5`, 
              2}, {3.077, 3}}, {{2.5, 2}, {3.655, 2}}, {{3.077, 1}, {1.923`, 
              1}}, {{3.077, 3}, {3.655, 2}}, {{3.655, 2}, {3.077, 1}}};
      ext = {{{1.345, 2}, {1.057, 1.5}}, {{1.345, 2}, {1.057, 2.5}}, {{1.345, 
              2}, {0.768, 2}}, {{1.923, 3}, {2.211, 3.5`}}, {{1.923, 
              3}, {1.634, 3.5}}, {{1.9223, 3}, {1.345, 3}}, {{3.077, 
              3}, {2.789, 3.5}}, {{3.077, 3}, {3.655, 3}}, {{3.077, 
              3}, {3.366, 3.5}}, {{3.655, 2}, {3.943, 2.5}}, {{3.655, 
              2}, {4.232, 2}}, {{3.655, 2}, {3.943, 1.5}}, {{3.077, 
              1}, {3.366, 0.5}}, {{3.077, 1}, {3.655, 1}}, {{3.077, 
              1}, {2.789, 0.5}}, {{1.923, 1}, {2.211, 0.5}}, {{1.923, 
              1}, {1.634, 0.5}}, {{1.923, 1}, {1.345, 1}}};
      frame = Union[ext, main];
      getR[m_] := 
        Module[{emptyList, emptyCrList, myA, myB, myCr}, Off[];
          emptyList = Table[0, {0}];
          emptyCrList = Table[0, {0}];
          While[True, {myA = RandomReal[{1, 4}];
              myB = RandomReal[{1, 3}];
              {cr1, cr2, cr3, 
                  cr4} = {(myB - 1.732*myA) < -0.33, (myB + 1.732*myA) < 
                    8.33, (myB + 1.732*myA) > 
                    4.330, (myB - 1.732*myA) > -4.330};              
              If[cr1 && cr2 && cr3 && 
                  cr4, {emptyList = Append[emptyList, {myA, myB}];
                  emptyCrList = Append[emptyCrList, myCr];}];
              If[Length[emptyList] == m, {Break[];}];}];
          Return[Transpose[{emptyList}]];];
      absords = Flatten[getR[n], 1];
      data = Transpose[{absords, RandomReal[{0, Pi}, n]}];
      lines = (Function[s, 
                  First[#] + 
                    s r/2 Through[{Cos, Sin}[Last[#]]]] /@ {1, -1}) & /@data;
      isIntersect[x1_, y1_, x2_, y2_, x3_, y3_, x4_, y4_] := 
        Module[{ua, ub, myR}, ua = 0; ub = 0; myR = 0;          
          If[(((y4 - y3)*(x2 - x1)) - ((x4 - x3)*(y2 - y1))) != 
              0, {ua = (((x4 - x3)*(y1 - y3)) - ((y4 - y3)*(x1 - x3)))/(((y4 -
                               y3)*(x2 - x1)) - ((x4 - x3)*(y2 - y1)));              
              ub = (((x2 - x1)*(y1 - y3)) - ((y2 - y1)*(x1 - x3)))/(((y4 - 
                              y3)*(x2 - x1)) - ((x4 - x3)*(y2 - y1)));
              If[(0 < ua < 1) && (0 < ub < 1), {myR = 1;}];}];
          Return[myR];];
      lands = 
        Table[isIntersect[lines[[i, 1, 1]], lines[[i, 1, 2]], 
            lines[[i, 2, 1]], lines[[i, 2, 2]], frame[[j, 1, 1]], 
            frame[[j, 1, 2]], frame[[j, 2, 1]], frame[[j, 2, 2]]], {j, 1, 
            Length[frame]}, {i, 1, n}];
      landnumbers = Table[Count[Transpose[lands][[i]], 1], {i, 1, n}];
      {hit0, hit1, hit2, hit3} = Table[Position[landnumbers, i], {i, 0, 3}];
      probs = {1 + r^2/2 - 3/2*r*(4 - N[Sqrt[3]]/2r)*t, -5/4*(r^2) + 
            3/2*r*(4 - N[Sqrt[3]]/2r)*t, (r^2) - 
            3*N[Sqrt[3]]/4*(r^2)*t, -r^2/4 + 3*N[Sqrt[3]]/4(r^2)*t};
      esti = N[((3*r*(8 - Sqrt[3]*r)/2))/(3 - 2*(Length[hit0]/n)), 15];
      wp = 1/(55 + Log[n^4]);
      esti
      ];
End[]
EndPackage[]


test2 = Table[DoubleGrid[100,1], 100];
Directory[];
Export["doublegrid_100.csv", test2, "CSV"];


test3 = Table[DoubleGrid[1000,1], 100];
Directory[];
Export["doublegrid_1000.csv", test3, "CSV"];


test4 = Table[DoubleGrid[10000,1], 100];
Directory[];
Export["doublegrid_10000-4.csv", test4, "CSV"];


test5 = Table[TripleGrid[100,1], 100];
Directory[];
Export["triplegrid_100-1.csv", test5, "CSV"];


test1 = Table[SingleGrid[100,1], 100];
Directory[];
Export["singlegrid_100-1.csv", test1, "CSV"];

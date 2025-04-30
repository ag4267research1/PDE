// mesh_ellipse_circle.geo
// 1) Characteristic mesh size
lc = 0.1; //smaller lc finer mesh

// 2) Geometry parameters
R   = 2.0;    // outer circle radius
a   = 1.0;    // inner ellipse semi-major axis
b   = 0.7;    // inner ellipse semi-minor axis
num = 200;    // points to approximate the ellipse

// 3) Outer circle (4 quarter-arcs)
Point(1) = { R,  0, 0, lc};
Point(2) = { 0,  R, 0, lc};
Point(3) = {-R,  0, 0, lc};
Point(4) = { 0, -R, 0, lc};
Point(0) = { 0,  0, 0, lc}; // center for arcs

Circle(   1) = {1, 0, 2};
Circle(   2) = {2, 0, 3};
Circle(   3) = {3, 0, 4};
Circle(   4) = {4, 0, 1};
Line Loop(10) = {1,2,3,4};

// 4) Inner ellipse (approximate by spline)
EllipsePoints[] = {};
For i In {0:num-1}
  xi = a*Cos(2*Pi*i/num);
  yi = b*Sin(2*Pi*i/num);
  pid = 1000 + i;
  Point(pid) = {xi, yi, 0, lc};
  EllipsePoints[i+1] = pid;
EndFor
// // after filling EllipsePoints[1..num]:
// EllipsePoints[num+1] = EllipsePoints[1];
// EllipsePoints[num+2] = EllipsePoints[2];
// EllipsePoints[num+3] = EllipsePoints[3];
// // now build a *periodic* cubic spline
// Spline(30) = EllipsePoints[];
// Line Loop(40) = {30};

// // after defining the array EllipsePoints[1..num]:
// For i In {1:num}
//     j = (i % num) + 1;              // next index, wrapping around
//     Line(1000 + i) = {EllipsePoints[i], EllipsePoints[j]};
//   EndFor
//   Line Loop(40) = {1001,1002,…,1000+num};

// // 5) Annular surface = outer minus inner
// Build straight line segments between successive points
EllipseLines[] = {};
For i In {1:num}
  j = (i % num) + 1;                // next index (wraps back to 1)
  lid = 2000 + i;                   // unique line ID
  Line(lid) = { EllipsePoints[i], EllipsePoints[j] };
  EllipseLines[i] = lid;            // collect into array
EndFor

// Now define the loop out of those lines
Line Loop(40) = EllipseLines[];

// And the annular surface as before

Plane Surface(60) = {10, 40};

// 6) Physical tags
Physical Curve("OuterCircle")      = {1,2,3,4};
Physical Curve("InnerEllipse")    = {40};
Physical Surface("AnnulusDomain") = {60};


//gmsh -2 mesh_ellipse_circle.geo -format msh2 -o mesh_ellipse_circle.msh
//meshio convert mesh_ellipse_circle.msh mesh_ellipse_circle.xdmf


// something thats compactly written 

/*
if (i < num)
  j = i + 1;
else
  j = 1;
endif
///////SAME AS 
j = (i % num) + 1;

*/
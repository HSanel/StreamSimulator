syms ux uy r0 r1 r2 r3 r4 r5 f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 real
syms f_t0 f_t1 f_t2 f_t3 f_t4 f_t5 f_t6 f_t7 f_t8 f_t9 real
syms Fx Fy rho real

x1=0-ux;
y1=0-uy;

x2=1-ux;
y2=0-uy;

x3=0-ux;
y3=1-uy;

x4=-1-ux;
y4=0 -uy;

x5=0 -ux;
y5=-1-uy;

x6=1-ux;
y6=1-uy;

x7=-1-ux;
y7=1 -uy;
		
x8=-1-ux;
y8=-1-uy;
		
x9=1 -ux;
y9=-1-uy;

M=[1,1,1,1,1,1,1,1,1;
x1,x2,x3,x4,x5,x6,x7,x8,x9;
y1,y2,y3,y4,y5,y6,y7,y8,y9;
x1*x1 + y1*y1, x2*x2 + y2*y2, x3*x3 + y3*y3, x4*x4 + y4*y4, x5*x5 + y5*y5, x6*x6 + y6*y6, x7*x7 + y7*y7, x8*x8 + y8*y8, x9*x9 + y9*y9 ;
x1*x1 - y1*y1, x2*x2 - y2*y2, x3*x3 - y3*y3, x4*x4 - y4*y4, x5*x5 - y5*y5, x6*x6 - y6*y6, x7*x7 - y7*y7, x8*x8 - y8*y8, x9*x9 - y9*y9 ;
x1*y1, x2*y2, x3*y3, x4*y4, x5*y5, x6*y6, x7*y7, x8*y8, x9*y9 ;
x1*x1*y1, x2*x2*y2, x3*x3*y3, x4*x4*y4, x5*x5*y5, x6*x6*y6, x7*x7*y7, x8*x8*y8, x9*x9*y9 ;
x1*y1*y1, x2*y2*y2, x3*y3*y3, x4*y4*y4, x5*y5*y5, x6*y6*y6, x7*y7*y7, x8*y8*y8, x9*y9*y9 ;
x1*x1 * y1*y1, x2*x2 * y2*y2, x3*x3 * y3*y3, x4*x4 * y4*y4, x5*x5 * y5*y5, x6*x6 * y6*y6, x7*x7 * y7*y7, x8*x8 * y8*y8, x9*x9 * y9*y9 ];

I=[1, 0, 0, 0, 0, 0, 0, 0, 0;
0, 1, 0, 0, 0, 0, 0, 0, 0;
0, 0, 1, 0, 0, 0, 0, 0, 0;
0, 0, 0, 1, 0, 0, 0, 0, 0;
0, 0, 0, 0, 1, 0, 0, 0, 0;
0, 0, 0, 0, 0, 1, 0, 0, 0;
0, 0, 0, 0, 0, 0, 1, 0, 0;
0, 0, 0, 0, 0, 0, 0, 1, 0;
0, 0, 0, 0, 0, 0, 0, 0, 1];

D=[r0, 0,  0,  0,  0,  0,  0,  0,  0 ;
0,  r1, 0,  0,  0,  0,  0,  0,  0 ;
0,  0,  r1, 0,  0,  0,  0,  0,  0 ;
0,  0,  0,  r2, 0,  0,  0,  0,  0 ;
0,  0,  0,  0,  r2, 0,  0,  0,  0 ;
0,  0,  0,  0,  0,  r3, 0,  0,  0 ;
0,  0,  0,  0,  0,  0,  r4, 0,  0 ;
0,  0,  0,  0,  0,  0,  0,  r4, 0 ;
0,  0,  0,  0,  0,  0,  0,  0,  r5];

cs_sq = 1.0/3.0;

f_dist=[f0, f1, f2, f3, f4, f5, f6, f7, f8];

m_eq=[rho, 0, 0, 2*rho*cs_sq, 0, 0, 0, 0, rho*cs_sq*cs_sq];

R=[0, Fx, Fy, 0, 0, 0, Fx*cs_sq, Fy*cs_sq,0];

f_dist=f_dist';
m_eq = m_eq';
R=R';

M_inv=inv(M);
f_tilde = D*(M*f_dist-m_eq);
f_r = (I-D/2)*R;

fileID = fopen('2DFormel.txt','w');

fprintf(fileID,'%s\n\n', char(ccode(f_tilde)));

f_tilde=[f_t0, f_t1, f_t2, f_t3, f_t4, f_t5, f_t6, f_t7, f_t8];
f_tilde = f_tilde';

Omega=-M_inv*f_tilde;
G=M_inv*f_r;

fprintf(fileID,'%s\n\n', char(ccode(Omega)));
fprintf(fileID,'%s\n\n', char(ccode(G)));


%f_star = f_dist + Omega + G;
fclose(fileID);

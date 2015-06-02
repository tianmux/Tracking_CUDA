#include "Header.h"

/*Update the momentum*/
double3 p_functionHost(double3 position, double3 momentum, double3 Efield, double3 Bfield, double t, double fre, double phi0)
{
	double3 dpodt;
	double qe = 1.6e-19;
	double Vc = 299792458.0;
	double me = 9.10938291e-31;

	Efield.x = Efield.x*sin(2.0*M_PI*fre*t + phi0);
	Efield.y = Efield.y*sin(2.0*M_PI*fre*t + phi0);
	Efield.z = Efield.z*sin(2.0*M_PI*fre*t + phi0);
	Bfield.x = -Bfield.x*sin(2.0*M_PI*fre*t + M_PI / 2.0 + phi0);
	Bfield.y = -Bfield.y*sin(2.0*M_PI*fre*t + M_PI / 2.0 + phi0);
	Bfield.z = -Bfield.z*sin(2.0*M_PI*fre*t + M_PI / 2.0 + phi0);

	dpodt.x = -qe*(Efield.x + Vc*(momentum.y*Bfield.z - momentum.z*Bfield.y)
		/ sqrt(momentum.x*momentum.x + momentum.y*momentum.y + momentum.z*momentum.z + me*me*Vc*Vc));
	dpodt.y = -qe*(Efield.y + Vc*(-momentum.x*Bfield.z + momentum.z*Bfield.x)
		/ sqrt(momentum.x*momentum.x + momentum.y*momentum.y + momentum.z*momentum.z + me*me*Vc*Vc));
	dpodt.z = -qe*(Efield.z + Vc*(momentum.x*Bfield.y - momentum.y*Bfield.x)
		/ sqrt(momentum.x*momentum.x + momentum.y*momentum.y + momentum.z*momentum.z + me*me*Vc*Vc));
	return dpodt;
}
void intersectHost(double3 point1, double3 point2, double3 point3, double3 pointA, double3 pointB, double *pointintpara)
{
	double a, b, c, d;//parameter of plane

	double ndotu;
	/*for plane*/
	a = (point2.y - point1.y) * (point3.z - point1.z) - (point3.y - point1.y)*(point2.z - point1.z);
	b = -((point2.x - point1.x) * (point3.z - point1.z) - (point3.x - point1.x)*(point2.z - point1.z));
	c = (point2.x - point1.x) * (point3.y - point1.y) - (point3.x - point1.x)*(point2.y - point1.y);
	d = -a*point1.x - b*point1.y - c*point1.z;
	/*to find the t of the intersection point */
	//http://geomalgorithms.com/a05-_intersect-1.html
	ndotu = a*(pointB.x - pointA.x) + b*(pointB.y - pointA.y) + c*(pointB.z - pointA.z);
	if (ndotu != 0)
	{
		*pointintpara = -(a*pointA.x + b*pointA.y + c*pointA.z + d) / ndotu;
	}
	else
		*pointintpara = 1e6;
}
void alpha_betaHost(double3 point1, double3 point2, double3 point3, double3 point, double2 *ab)
{
	/*Needed some trick to do the determint, assumed a fake third vector that gives {1,2,3} in the third row in parameter matrix, corresponding to a fake gama variable*/
	(*ab).x = ((point.x - point1.x)*((point3.y - point1.y) * 3 - (point3.z - point1.z) * 2) - (point3.x - point1.x)*((point.y - point1.y) * 3 - (point.z - point1.z) * 2) + 1 * ((point.y - point1.y)*(point3.z - point1.z) - (point.z - point1.z)*(point3.y - point1.y))) /
		((point2.x - point1.x)*((point3.y - point1.y) * 3 - (point3.z - point1.z) * 2) - (point3.x - point1.x)*((point2.y - point1.y) * 3 - (point2.z - point1.z) * 2) + 1 * ((point2.y - point1.y)*(point3.z - point1.z) - (point2.z - point1.z)*(point3.y - point1.y)));
	(*ab).y = ((point2.x - point1.x)*((point.y - point1.y) * 3 - (point.z - point1.z) * 2) - (point.x - point1.x)*((point2.y - point1.y) * 3 - (point2.z - point1.z) * 2) + 1 * ((point2.y - point1.y)*(point.z - point1.z) - (point2.z - point1.z)*(point.y - point1.y))) /
		((point2.x - point1.x)*((point3.y - point1.y) * 3 - (point3.z - point1.z) * 2) - (point3.x - point1.x)*((point2.y - point1.y) * 3 - (point2.z - point1.z) * 2) + 1 * ((point2.y - point1.y)*(point3.z - point1.z) - (point2.z - point1.z)*(point3.y - point1.y)));


}
void tetvolumeHost(double3 point1, double3 point2, double3 point3, double3 point4, double *volume)
{
	*volume = (point4.x - point1.x)*((point2.y - point1.y)*(point3.z - point1.z) - (point3.y - point1.y)*(point2.z - point1.z)) -
		(point4.y - point1.y)*((point2.x - point1.x)*(point3.z - point1.z) - (point3.x - point1.x)*(point2.z - point1.z)) +
		(point4.z - point1.z)*((point2.x - point1.x)*(point3.y - point1.y) - (point3.x - point1.x)*(point2.y - point1.y));
}
void moveKernelHost(double4 *barycentric, double3 *position, double3 *momentum, double3* H_position0, double3* H_momentumf, double3* Efield, double3 *Bfield, double3* impactmomentum,
	double *nodes, double* volume,
	double norm, double t, double dt, double fre, double fdnorm,
	int* impact, int *meshindextet, int* tetmesh, int* flag, int* initphase,
	int tetsize, int N_par, int N_cycles, int phase_step)
{

	double Vc = 299792458.0;
	double me = 9.10938291e-31;
	double phi0;

	t = t / fre;
	dt = dt / fre;

	int i;
	int j;
	

#pragma omp parallel for private(j)
	for (i = 0; i < N_par;i++)
	{
		double3 dEfdtemp[4], dBfdtemp[4];// "d" means this is a device variable;
		double3 pEfd, pBfd;//fields at particle location
		double3 vertexs[4];
		double3 position0;
		double3 positionold;
		double3 momentumt, momentum0, momentum1, momentum2, momentum3, momentum4, momentumf;
		double3 v;
		int tempflag;
		int tetindex;
		tempflag = flag[i];
		if (flag[i] >= -1)
		{
			position0.x = position[i].x*norm;
			position0.y = position[i].y*norm;
			position0.z = position[i].z*norm;
			positionold.x = position0.x;
			positionold.y = position0.y;
			positionold.z = position0.z;

			momentum0.x = momentum[i].x;
			momentum0.y = momentum[i].y;
			momentum0.z = momentum[i].z;
			tetindex = meshindextet[i];

			/*get the field for four vertex of the tet where the particle was found*/
			for (j = 0; j < 4; j++)
			{
				dEfdtemp[j].x = Efield[tetmesh[tetindex*tetsize + j + 1]].x * fdnorm;
				dEfdtemp[j].y = Efield[tetmesh[tetindex*tetsize + j + 1]].y * fdnorm;
				dEfdtemp[j].z = Efield[tetmesh[tetindex*tetsize + j + 1]].z * fdnorm;
				dBfdtemp[j].x = Bfield[tetmesh[tetindex*tetsize + j + 1]].x * fdnorm;
				dBfdtemp[j].y = Bfield[tetmesh[tetindex*tetsize + j + 1]].y * fdnorm;
				dBfdtemp[j].z = Bfield[tetmesh[tetindex*tetsize + j + 1]].z * fdnorm;
				vertexs[j].x = nodes[tetmesh[tetindex*tetsize + j + 1] * 3];
				vertexs[j].y = nodes[tetmesh[tetindex*tetsize + j + 1] * 3 + 1];
				vertexs[j].z = nodes[tetmesh[tetindex*tetsize + j + 1] * 3 + 2];
			}
			pEfd.x = barycentric[i].x*dEfdtemp[3].x + barycentric[i].y*dEfdtemp[1].x + barycentric[i].z*dEfdtemp[2].x + barycentric[i].w*dEfdtemp[0].x;
			pEfd.y = barycentric[i].x*dEfdtemp[3].y + barycentric[i].y*dEfdtemp[1].y + barycentric[i].z*dEfdtemp[2].y + barycentric[i].w*dEfdtemp[0].y;
			pEfd.z = barycentric[i].x*dEfdtemp[3].z + barycentric[i].y*dEfdtemp[1].z + barycentric[i].z*dEfdtemp[2].z + barycentric[i].w*dEfdtemp[0].z;
			pBfd.x = barycentric[i].x*dBfdtemp[3].x + barycentric[i].y*dBfdtemp[1].x + barycentric[i].z*dBfdtemp[2].x + barycentric[i].w*dBfdtemp[0].x;
			pBfd.y = barycentric[i].x*dBfdtemp[3].y + barycentric[i].y*dBfdtemp[1].y + barycentric[i].z*dBfdtemp[2].y + barycentric[i].w*dBfdtemp[0].y;
			pBfd.z = barycentric[i].x*dBfdtemp[3].z + barycentric[i].y*dBfdtemp[1].z + barycentric[i].z*dBfdtemp[2].z + barycentric[i].w*dBfdtemp[0].z;
			phi0 = (double)(initphase[i]) / phase_step*M_PI;
			//Runge-Kutta method to find the next momentum.
			momentumf = p_functionHost(position0, momentum0, pEfd, pBfd, t, fre, phi0);
			momentum1.x = momentumf.x*dt;
			momentum1.y = momentumf.y*dt;
			momentum1.z = momentumf.z*dt;

			momentumt.x = momentum0.x + momentum1.x / 2.0;
			momentumt.y = momentum0.y + momentum1.y / 2.0;
			momentumt.z = momentum0.z + momentum1.z / 2.0;
			momentumf = p_functionHost(position0, momentumt, pEfd, pBfd, t + dt / 2.0, fre, phi0);
			momentum2.x = momentumf.x*dt;
			momentum2.y = momentumf.y*dt;
			momentum2.z = momentumf.z*dt;

			momentumt.x = momentum0.x + momentum2.x / 2.0;
			momentumt.y = momentum0.y + momentum2.y / 2.0;
			momentumt.z = momentum0.z + momentum2.z / 2.0;
			momentumf = p_functionHost(position0, momentumt, pEfd, pBfd, t + dt / 2.0, fre, phi0);
			momentum3.x = momentumf.x*dt;
			momentum3.y = momentumf.y*dt;
			momentum3.z = momentumf.z*dt;

			momentumt.x = momentum0.x + momentum3.x / 2.0;
			momentumt.y = momentum0.y + momentum3.y / 2.0;
			momentumt.z = momentum0.z + momentum3.z / 2.0;
			momentumf = p_functionHost(position0, momentumt, pEfd, pBfd, t + dt, fre, phi0);
			momentum4.x = momentumf.x*dt;
			momentum4.y = momentumf.y*dt;
			momentum4.z = momentumf.z*dt;

			momentumf.x = momentum0.x + momentum1.x / 6.0 + momentum2.x / 3.0 + momentum3.x / 3.0 + momentum4.x / 6.0;
			momentumf.y = momentum0.y + momentum1.y / 6.0 + momentum2.y / 3.0 + momentum3.y / 3.0 + momentum4.y / 6.0;
			momentumf.z = momentum0.z + momentum1.z / 6.0 + momentum2.z / 3.0 + momentum3.z / 3.0 + momentum4.z / 6.0;

			//Compute new velocity:

			v.x = Vc*momentumf.x / sqrt(me*me*Vc*Vc + momentumf.x*momentumf.x + momentumf.y*momentumf.y + momentumf.z*momentumf.z);
			v.y = Vc*momentumf.y / sqrt(me*me*Vc*Vc + momentumf.x*momentumf.x + momentumf.y*momentumf.y + momentumf.z*momentumf.z);
			v.z = Vc*momentumf.z / sqrt(me*me*Vc*Vc + momentumf.x*momentumf.x + momentumf.y*momentumf.y + momentumf.z*momentumf.z);

			//Compute new position;
			position0.x += v.x*dt;
			position0.y += v.y*dt;
			position0.z += v.z*dt;
			//put the temporary new position and momentum into global mem for later use. 
			H_position0[i].x = position0.x;
			H_position0[i].y = position0.y;
			H_position0[i].z = position0.z;

			H_momentumf[i].x = momentumf.x;
			H_momentumf[i].y = momentumf.y;
			H_momentumf[i].z = momentumf.z;

		}

	}
}
//locate the particle in tet mesh
void trackingHost(double4 *barycentric, double3 *Hposition, double3 *H_position0, double3 *momentum, double3* H_momentumf, double3* impactmomentum,
	double norm, double *nodes, double *volume,
	int *meshindextet, int *oldmesh, int* flag, int* impact,
	int N_par, int N_cycles, int tetsize)
{
	int i;
	int j;//iteration index.
	
#pragma omp parallel for private(j)
	for (i = 0; i < N_par;i++)
	{
		int tetindex;
		int count = 0;//if search for 10 tet and didn't find, the count as lost
		int tempflag;
		int found = 0;//flag indicating that we didn't find the tet that contain the particle yet.
		double tempvolume;
		double a, b, c, d;//the Barycentric coordinates of particle in tet;
		double interpara[4];//the parameters for intersections of trajectory with each face of tet

		double2 ab[4];
		double3 nodePosition[4];//node position of one tet
		double3 intersec[4];//the intersection point on each plane of faces of a tet
		double3 pold, pnew;
		found = 0;
		count = 0;
		pold.x = Hposition[i].x*norm;
		pold.y = Hposition[i].y*norm;
		pold.z = Hposition[i].z*norm;
		pnew.x = H_position0[i].x;
		pnew.y = H_position0[i].y;
		pnew.z = H_position0[i].z;
		tempflag = flag[i];
		tetindex = meshindextet[i];
		while (found == 0 && count <= 10)
		{
			count++;

			for (j = 0; j < 4; j++)
			{
				interpara[j] = 0;
				ab[j].x = 1e6;
				ab[j].y = 1e6;
			}
			//get the nodes of the tet that might contain the new position of particle, start from the old one
			for (j = 0; j < 4; j++)
			{
				nodePosition[j].x = nodes[oldmesh[tetindex*tetsize + 1 + j] * 3];
				nodePosition[j].y = nodes[oldmesh[tetindex*tetsize + 1 + j] * 3 + 1];
				nodePosition[j].z = nodes[oldmesh[tetindex*tetsize + 1 + j] * 3 + 2];
			}

			//get the signed volumes of tet, and calculate volume of 123P, 243P, 341P, 421P, in order to make the order right for surface recgonition we need to change the order 
			//012 cooresponding to face shared with first neighbor
			//123 cooresponding to face shared with fourth neighbor
			//230 cooresponding to face shared with second neighbor
			//310 cooresponding to face shared with third neighbor
			tempvolume = volume[tetindex];
			tetvolumeHost(nodePosition[0], nodePosition[1], nodePosition[2], pnew, &a);
			a = a / tempvolume;
			tetvolumeHost(nodePosition[2], nodePosition[3], nodePosition[0], pnew, &b);
			b = b / tempvolume;
			tetvolumeHost(nodePosition[3], nodePosition[1], nodePosition[0], pnew, &c);
			c = c / tempvolume;
			tetvolumeHost(nodePosition[1], nodePosition[3], nodePosition[2], pnew, &d);
			d = d / tempvolume;


			if (a >= -1e-15 && b >= -1e-15 && c >= -1e-15 && d >= -1e-15)
			{
				found = 1;
				barycentric[i].x = a;
				barycentric[i].y = b;
				barycentric[i].z = c;
				barycentric[i].w = d;
				tempflag = 0;
			}

			else
			{
				if (a < -1e-15)
				{
					intersectHost(nodePosition[0], nodePosition[1], nodePosition[2], pold, pnew, &interpara[0]);
					intersec[0].x = pold.x + interpara[0] * (pnew.x - pold.x);
					intersec[0].y = pold.y + interpara[0] * (pnew.y - pold.y);
					intersec[0].z = pold.z + interpara[0] * (pnew.z - pold.z);
					alpha_betaHost(nodePosition[0], nodePosition[1], nodePosition[2], intersec[0], &ab[0]);
					/*if (tetindex == 66039)
					{
					found = 1;
					tempflag = tetindex;
					barycentric[i].x = ab[0].x;
					barycentric[i].y = ab[0].y;
					barycentric[i].z = intersec[0].z;
					return;
					}*/

				}
				if (b < -1e-15)
				{
					intersectHost(nodePosition[2], nodePosition[3], nodePosition[0], pold, pnew, &interpara[1]);
					intersec[1].x = pold.x + interpara[1] * (pnew.x - pold.x);
					intersec[1].y = pold.y + interpara[1] * (pnew.y - pold.y);
					intersec[1].z = pold.z + interpara[1] * (pnew.z - pold.z);
					alpha_betaHost(nodePosition[2], nodePosition[3], nodePosition[0], intersec[1], &ab[1]);
					/*if (tetindex == 66039)
					{
					found = 1;
					tempflag = tetindex+1;
					barycentric[i].x = ab[1].x;
					barycentric[i].y = ab[1].y;
					barycentric[i].z = intersec[1].z;
					return;
					}*/
				}
				if (c < -1e-15)
				{
					intersectHost(nodePosition[3], nodePosition[1], nodePosition[0], pold, pnew, &interpara[2]);
					intersec[2].x = pold.x + interpara[2] * (pnew.x - pold.x);
					intersec[2].y = pold.y + interpara[2] * (pnew.y - pold.y);
					intersec[2].z = pold.z + interpara[2] * (pnew.z - pold.z);
					alpha_betaHost(nodePosition[3], nodePosition[1], nodePosition[0], intersec[2], &ab[2]);

					/*if (tetindex == 66039)
					{
					found = 1;
					tempflag = tetindex+2;
					barycentric[i].x = ab[2].x;
					barycentric[i].y = ab[2].y;
					barycentric[i].z = intersec[2].z;
					return;
					}*/
				}
				if (d < -1e-15)
				{
					intersectHost(nodePosition[1], nodePosition[3], nodePosition[2], pold, pnew, &interpara[3]);
					intersec[3].x = pold.x + interpara[3] * (pnew.x - pold.x);
					intersec[3].y = pold.y + interpara[3] * (pnew.y - pold.y);
					intersec[3].z = pold.z + interpara[3] * (pnew.z - pold.z);
					alpha_betaHost(nodePosition[1], nodePosition[3], nodePosition[2], intersec[3], &ab[3]);
					/*if (tetindex == 66039)
					{
					found = 1;
					tempflag = tetindex+3;
					barycentric[i].x = nodePosition[1].x;
					barycentric[i].y = nodePosition[1].y;
					barycentric[i].z = nodePosition[1].z;
					return;
					}*/
				}
				for (j = 0; j < 4; j++)
				{
					if (ab[j].x <= 1 && ab[j].x >= -1e-15 && ab[j].y <= 1 && ab[j].y > -1e-15 && ab[j].x + ab[j].y <= 1)
					{
						/*if (tetindex == 66039)
						{
						found = 1;
						tempflag = tetindex+j+10;
						barycentric[i].x = ab[j].x;
						barycentric[i].y = ab[j].y;
						barycentric[i].z = intersec[0].z;
						return;
						}*/
						if (oldmesh[tetindex*tetsize + j + 5] != -1)
						{
							if (tempflag > -1)
							{
								found = 1;
								barycentric[i].x = a > 0 ? a : 0;
								barycentric[i].y = b > 0 ? b : 0;
								barycentric[i].z = c > 0 ? c : 0;
								barycentric[i].w = d > 0 ? d : 0;
								tempflag = -1;

								impactmomentum[i*N_cycles * 2 + impact[i]].x = H_momentumf[i].x;
								impactmomentum[i*N_cycles * 2 + impact[i]].y = H_momentumf[i].y;
								impactmomentum[i*N_cycles * 2 + impact[i]].z = H_momentumf[i].z;
								impact[i] ++;

								pnew.x = pold.x + interpara[j] * (pnew.x - pold.x);
								pnew.y = pold.y + interpara[j] * (pnew.y - pold.y);
								pnew.z = pold.z + interpara[j] * (pnew.z - pold.z);

								H_momentumf[i].x = 0;
								H_momentumf[i].y = 0;
								H_momentumf[i].z = 0;
								momentum[i].x = 0;
								momentum[i].y = 0;
								momentum[i].z = 0;
								j = 5;
							}
							else
							{
								found = 1;
								tempflag = -2;//means the particle is dead
								barycentric[i].x = 0;
								barycentric[i].y = 0;
								barycentric[i].z = 0;
								barycentric[i].w = 0;

								pnew.x = 0;
								pnew.y = 0;
								pnew.z = 0;
								H_momentumf[i].x = 0;
								H_momentumf[i].y = 0;
								H_momentumf[i].z = 0;
								momentum[i].x = 0;
								momentum[i].y = 0;
								momentum[i].z = 0;
								j = 5;
							}

						}
						else
						{
							tetindex = oldmesh[tetindex*tetsize + j + 10];//if particle hits a shared wall, then it moves to the neighbor tet
							j = 5;
						}
					}
				}
			}
			if (count == 10)
			{
				found = 1;
				tempflag = -3;//means the particle is lost
				barycentric[i].x = 0;
				barycentric[i].y = 0;
				barycentric[i].z = 0;
				barycentric[i].w = 0;

				pnew.x = 0;
				pnew.y = 0;
				pnew.z = 0;

				H_momentumf[i].x = 0;
				H_momentumf[i].y = 0;
				H_momentumf[i].z = 0;
				momentum[i].x = 0;
				momentum[i].y = 0;
				momentum[i].z = 0;
			}
		}
		Hposition[i].x = pnew.x / norm;
		Hposition[i].y = pnew.y / norm;
		Hposition[i].z = pnew.z / norm;
		momentum[i].x = H_momentumf[i].x;
		momentum[i].y = H_momentumf[i].y;
		momentum[i].z = H_momentumf[i].z;
		meshindextet[i] = tetindex;
		flag[i] = tempflag;
	}
	

}

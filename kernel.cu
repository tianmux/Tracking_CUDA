/*
1. Create two VBO, one for position one for velocity of particles
2. Create two VBO, one for triangles drawing one for edge drawing
3. Register the VBOs with Cuda
4. Map the VBO for writing from Cuda
5. Run Cuda kernel to modify the vertex positions
6. Unmap the VBO
7. Render the results using OpenGL
*/

#include"Header.h"
// includes, cuda
#include <cuda_gl_interop.h>
#include <curand_kernel.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop
#include <cuda_runtime.h>
#include <cub/cub/cub.cuh>

#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.30f
#define REFRESH_DELAY     10 //ms


/* This is the name of the data file we will read. */
std::string modelfile = "56realport.ncdf";//"mod0.mod"
std::string  fieldfile = "56realport.mod";
const char* FILE_NAME = modelfile.c_str();
const char* FILE_NAME2 = fieldfile.c_str();

/* Handle errors by printing an error message and exiting with a
* non-zero status. */
#define ERRCODE 2
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); /*exit(ERRCODE);*/}


////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width = 1024;
const unsigned int window_height = 1024;
clock_t timer0;
int N_par = 1;
int N_cycles = 2;//number of RF cycles we are going to track the particles
double fre;
double Vc = 299792458.0;
double epsilon = 8.854187817e-12;
double mu = 1.2566370614e-6;
double me = 9.1e-31;
double qe = 1.60217657e-19;


//Variables
int isGPU = 1;
std::map<std::string, std::string> inputs;//inputs from input file
double3 Efield, Bfield;//field info on every nodes
double3 *Efd_temp, *Bfd_temp;
double3 *dEfd, *dBfd;
double3 *D_p_Efd, *D_p_Bfd;//store the field at the location of particle
double3* impactmomentum;
double3* dimpactmomentum;
double* impactenergy;
double* impactenergy_shuffle;
double* D_impactenergy;

cudaError_t err;
double xrange[2] = { 0.117, 0.179 }, yrange[2] = { -0.19, -0.103 }, zrange[2] = { 1.574, 1.74 };

double fdnorm = 1e6;//normalize of field 
double fdnorm_max = 1e8;
double fdnorm_step = 1e6;
double initenergy = 2;//initial energy of seconday electrons, eV
double *Efd, *Bfd;//field info from file
double *Efd_img, *Bfd_img;

int* impact, *impact_shuffle;//number of impact
int* dimpact;


int* flag, *flag_shuffle;//indicate whether the particle is dead or live. 0 means live, 1 means just experinced an impact, -2 means dead, 
int* dflag;
double dt = 0.01;
int* initphase,*initphase_shuffle;
int* dinitphase;
int phase_step = 2;

double3 *Hposition, *Hvelocity;//position and momentum stored in host
double3 *Hposition_shuffle, *Hvelocity_shuffle;//position and momentum used for shuffle the dead particle out
double3 *d_position0, *d_momentumf;//store the intermediant position and momentum of particle during the tracking
double3 *H_position0, *H_momentumf, *H_position0_shuffle, *H_momentumf_shuffle;
double3 *D_momentumt;//store the intermediant momentum for Runge-Kutta steps
double4 *barycentric, *barycentric_shuffle;//the barycentric coordinates of the particle in each tet;four compoments are representing the coordinate coefficients corresponding to 4,1,2,3 vertex.
double4 *dbarycentric;
double *nodes;
double *dnodes;//nodes info for device use
double *mincor, *maxcor;
double *nodesdisp;//nodes for display
double *volume;//store the signed volume of the tets;
double *dvolume;

int *tetext;//node structure of exterior tets
int *tetint;//node structure of internal tets
int *tetall;//node structure of all tets, to be used in GPU
int *dtetall;
int4 *D_p_nodes;//store the temp info of the nodes of tet where the particle is located

int *meshindextet,*meshindextet_shuffle;// tet mesh, use for straight first then flush it with tet mesh info
int *meshindextet_temp;//use to store the active tetmesh temporarily, then we move the info to meshindextet
int *dmeshindextet;

int numactive;//the number of tet that is active(in the xyz range)


int *indexes;//the array stores the indexes of external tets
int *indexesedge;//the array stores the indexes of external surfaces of external tets.
unsigned int ntriangles;//number of triangles of exterior tetrahedrons
unsigned int nnodes, next,nint,nall;//number of total nodes, and number of exterior and interior tets


/*Generate the tet node structure, 14 entries:
first entry is the volume group id,
following four are node indexes,
next four are surface type(-1 means shared with neighbor, and other numbers represent the sideset id),
next one is the id where the tet is located in a easier to search straight mesh, the ID is calculated as sizeofx*sizeofy*Z+sizeofx*Y+X
next four are the neighbors cooresponding to the four sides. 
*/
int tetsize = 14;

double norm; //the normalize factor

struct diminfo
{
	char namedim[MAX_NC_NAME];//allocatet the array for dimension names
	size_t lengthp;//Length of the dimension
};

struct varinfo
{
	char namevar[MAX_NC_NAME];//name of the variable
	//#define	NC_NAT 	        0	/**< Not A Type */
	//#define	NC_BYTE         1	/**< signed 1 byte integer */
	//#define	NC_CHAR 	2	/**< ISO/ASCII character */
	//#define	NC_SHORT 	3	/**< signed 2 byte integer */
	//#define	NC_INT 	        4	/**< signed 4 byte integer */
	//#define NC_LONG         NC_INT  /**< deprecated, but required for backward compatibility. */
	//#define	NC_FLOAT 	5	/**< single precision floating point number */
	//#define	NC_DOUBLE 	6	/**< double precision floating point number */
	//#define	NC_UBYTE 	7	/**< unsigned 1 byte int */
	//#define	NC_USHORT 	8	/**< unsigned 2-byte int */
	//#define	NC_UINT 	9	/**< unsigned 4-byte int */
	//#define	NC_INT64 	10	/**< signed 8-byte int */
	//#define	NC_UINT64 	11	/**< unsigned 8-byte int */
	//#define	NC_STRING 	12	/**< string */
	nc_type typevar;//type of the variable
	int numdim;
	int *dimids = new int[NC_MAX_VAR_DIMS];//the id of dimensions that are used by this variable
	int numatt;//number of attributions
};

struct variablevalue
{
	int typevar;
	std::vector<int> value4;
	std::vector<double> value6;
};
int retval;//error code

// vbo variables
GLuint vbo, vbov;/*vbo is position, vbov is velocity*/
GLuint nvbo;//the vbo of nodes
GLuint vboindex;/*vboindex is the index of vertex*/
GLuint vboindexedg;/*vbo object for index of node to generate edges*/

struct cudaGraphicsResource *cuda_vbo_resource, *cuda_vbov_resource;
void *d_vbo_buffer = NULL;
void *d_vbov_buffer = NULL;

double g_fAnim = 0.0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = -90.0;
float translate_z = -0.1;
float translate_x = 0;
float translate_y = 0.0;

StopWatchInterface *timer = NULL;

// Auto-Verification Code
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_bQAReadback = false;

int *pArgc = NULL;
char **pArgv = NULL;
const char *sSDKsample = "simpleGL (VBO)";

#define MAX(a,b) ((a > b) ? a : b)

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(int argc, char **argv, char *ref_file);
void cleanup();

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
	unsigned int vbo_res_flags);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

// Cuda functionality
void runCuda(struct cudaGraphicsResource **vbo_resource, struct cudaGraphicsResource **vbov_resource, int func);
void runAutoTest(int devID, char **argv, char *ref_file);
void checkResultCuda(int argc, char **argv, const GLuint &vbo);


/*Define functions*/
void tetvolumeHost(double3 point1, double3 point2, double3 point3, double3 point4, double *volume);
void ConnectTetHost_2(int* oldmesh, int tetsize, int nall, int nnodes);
void showcudaerror();
double3 p_functionHost(double3 position, double3 momentum, double3 Efield, double3 Bfield, double t, double fre, double phi0);
void intersectHost(double3 point1, double3 point2, double3 point3, double3 pointA, double3 pointB, double *pointintpara);
void alpha_betaHost(double3 point1, double3 point2, double3 point3, double3 point, double2 *ab);
void tetvolumeHost(double3 point1, double3 point2, double3 point3, double3 point4, double *volume);
void moveKernelHost(double4 *barycentric, double3 *position, double3 *momentum, double3* d_position0, double3* d_momentumf, double3* Efield, double3 *Bfield, double3* impactmomentum,
	double *nodes, double* volume,
	double norm, double t, double dt, double fre, double fdnorm,
	int* impact, int *meshindextet, int* tetmesh, int* flag, int* initphase,
	int tetsize, int N_par, int N_cycles, int phase_step);
void trackingHost(double4 *barycentric, double3 *positionold, double3 *positionnew, double3 *momentum, double3* d_momentumf, double3* impactmomentum,
	double norm, double *nodes, double *volume,
	int *meshindextet, int *oldmesh, int* flag, int* impact,
	int N_par, int N_cycles, int tetsize);

//Kernels
//calculate the signed volume of tet with four arbitrary vertex points. 
__device__ void tetvolume(double3 point1, double3 point2, double3 point3, double3 point4, double *volume)
{
	*volume = (point4.x - point1.x)*((point2.y - point1.y)*(point3.z - point1.z) - (point3.y - point1.y)*(point2.z - point1.z)) -
		(point4.y - point1.y)*((point2.x - point1.x)*(point3.z - point1.z) - (point3.x - point1.x)*(point2.z - point1.z)) +
		(point4.z - point1.z)*((point2.x - point1.x)*(point3.y - point1.y) - (point3.x - point1.x)*(point2.y - point1.y));
}
//calculate the intercept point of a plane(from three points) and a line(two points)
__device__ void intersect(double3 point1, double3 point2, double3 point3, double3 pointA, double3 pointB, double *pointintpara)
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
//find the alpha beta coordinate of intersection point on a triangle
__device__ void alpha_beta(double3 point1, double3 point2, double3 point3, double3 point, double2 *ab)
{
	/*Needed some trick to do the determint, assumed a fake third vector that gives {1,2,3} in the third row in parameter matrix, corresponding to a fake gama variable*/
	(*ab).x = ((point.x - point1.x)*((point3.y - point1.y) * 3 - (point3.z - point1.z) * 2) - (point3.x - point1.x)*((point.y - point1.y) * 3 - (point.z - point1.z) * 2) + 1 * ((point.y - point1.y)*(point3.z - point1.z) - (point.z - point1.z)*(point3.y - point1.y))) /
		((point2.x - point1.x)*((point3.y - point1.y) * 3 - (point3.z - point1.z) * 2) - (point3.x - point1.x)*((point2.y - point1.y) * 3 - (point2.z - point1.z) * 2) + 1 * ((point2.y - point1.y)*(point3.z - point1.z) - (point2.z - point1.z)*(point3.y - point1.y)));
	(*ab).y = ((point2.x - point1.x)*((point.y - point1.y) * 3 - (point.z - point1.z) * 2) - (point.x - point1.x)*((point2.y - point1.y) * 3 - (point2.z - point1.z) * 2) + 1 * ((point2.y - point1.y)*(point.z - point1.z) - (point2.z - point1.z)*(point.y - point1.y))) /
		((point2.x - point1.x)*((point3.y - point1.y) * 3 - (point3.z - point1.z) * 2) - (point3.x - point1.x)*((point2.y - point1.y) * 3 - (point2.z - point1.z) * 2) + 1 * ((point2.y - point1.y)*(point3.z - point1.z) - (point2.z - point1.z)*(point3.y - point1.y)));
}
//initailize the particles from the exterior surfaces
__global__ void initpar(double4 *barycentric, double3 *position, double3 *momentum, double3* d_position0, double3* d_momentumf, double3* D_momentumt,
	double* D_impactenergy,	double* nodes, double* volume,
	double norm,
	int* impact, int *meshindextet, int* flag, int* tetmesh, int*initphase,
	int N_par, int N_cycles, int tetsize, int phase_step)
{
	long int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j;//iteration index
	double3 vertexs[4];
	double a, b, c, d,tempvolume;
	int tempid;
	int tempmask[12] = { 0, 1, 2, 0, 2, 3, 0, 3, 2, 1, 2, 3 };

	/*Generate the particles on centers of the exterior triangles*/
	if (i<N_par)
	{
		tempid = meshindextet[i];
		for (j = 0; j < 4; j++)
		{
			vertexs[j].x = nodes[tetmesh[tempid*tetsize + 1 + j] * 3];
			vertexs[j].y = nodes[tetmesh[tempid*tetsize + 1 + j] * 3 + 1];
			vertexs[j].z = nodes[tetmesh[tempid*tetsize + 1 + j] * 3 + 2];
		}
		for (j = 0; j < 4; j++)
		{
			if (tetmesh[tempid*tetsize + 5 + j] != -1)
			{
				position[i].x = (vertexs[tempmask[j * 3]].x + vertexs[tempmask[j * 3 + 1]].x + vertexs[tempmask[j * 3 + 2]].x) / 3.0;
				position[i].y = (vertexs[tempmask[j * 3]].y + vertexs[tempmask[j * 3 + 1]].y + vertexs[tempmask[j * 3 + 2]].y) / 3.0;
				position[i].z = (vertexs[tempmask[j * 3]].z + vertexs[tempmask[j * 3 + 1]].z + vertexs[tempmask[j * 3 + 2]].z) / 3.0;

			}
		}
		d_position0[i].x = position[i].x;
		d_position0[i].y = position[i].y;
		d_position0[i].z = position[i].z;
		momentum[i].x = 0.0;
		momentum[i].y = 0.0;
		momentum[i].z = 0.0;
		d_momentumf[i].x = 0.0;
		d_momentumf[i].y = 0.0;
		d_momentumf[i].z = 0.0;
		D_momentumt[i].x = 0.0;
		D_momentumt[i].y = 0.0;
		D_momentumt[i].z = 0.0;
		impact[i] = 0;
		flag[i] = 0;
		for (j = 0; j < N_cycles*2; j++)
		{
			D_impactenergy[i*N_cycles*2 + j]= 0;
		}

		tempvolume = volume[tempid];
		tetvolume(vertexs[0], vertexs[1], vertexs[2], position[i], &a);
		barycentric[i].x = a / tempvolume;
		tetvolume(vertexs[2], vertexs[3], vertexs[0], position[i], &b);
		barycentric[i].y = b / tempvolume;
		tetvolume(vertexs[3], vertexs[1], vertexs[0], position[i], &c);
		barycentric[i].z = c / tempvolume;
		tetvolume(vertexs[1], vertexs[3], vertexs[2], position[i], &d);
		barycentric[i].w = d / tempvolume;

		position[i].x = position[i].x / norm;
		position[i].y = position[i].y / norm;
		position[i].z = position[i].z / norm;
		d_position0[i].x = d_position0[i].x;
		d_position0[i].y = d_position0[i].y;
		d_position0[i].z = d_position0[i].z;

		initphase[i] = i%phase_step;

		
	}
}
//locate the particle in tet mesh
__global__ void tracking(double4 *barycentric, double3 *positionold,  double3 *positionnew, double3 *momentum, double3* d_momentumf,
	double* D_impactenergy,double norm, double *nodes, double *volume,
	int *meshindextet, int *oldmesh, int* flag,  int* impact,
	int N_par, int N_cycles, int tetsize)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j;//iteration index.
	int tetindex;
	int count = 0;//if search for 10 tet and didn't find, the count as lost
	int tempflag;
	int found = 0;//flag indicating that we didn't find the tet that contain the particle yet.
	double tempvolume;
	double a, b, c, d;//the Barycentric coordinates of particle in tet;
	double interpara[4];//the parameters for intersections of trajectory with each face of tet
	double qe = 1.6e-19;
	double Vc = 299792458.0;
	double me = 9.10938291e-31;
	double2 ab[4];
	double3 nodePosition[4];//node position of one tet
	double3 intersec[4];//the intersection point on each plane of faces of a tet
	double3 pold, pnew;

	found = 0;
	if (i < N_par)
	{
		pold.x = positionold[i].x*norm;
		pold.y = positionold[i].y*norm;
		pold.z = positionold[i].z*norm;
		pnew.x = positionnew[i].x;
		pnew.y = positionnew[i].y;
		pnew.z = positionnew[i].z;
		tempflag = flag[i];
		tetindex = meshindextet[i];
		if (tempflag < -1)
		{
			return;
		}
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
			tetvolume(nodePosition[0], nodePosition[1], nodePosition[2], pnew, &a);
			a = a / tempvolume;
			tetvolume(nodePosition[2], nodePosition[3], nodePosition[0], pnew, &b);
			b = b / tempvolume;
			tetvolume(nodePosition[3], nodePosition[1], nodePosition[0], pnew, &c);
			c = c / tempvolume;
			tetvolume(nodePosition[1], nodePosition[3], nodePosition[2], pnew, &d);
			d = d / tempvolume;


			if (a >= -1e-15 && b >= -1e-15 && c >= -1e-15 && d >= -1e-15)
			{
				found = 1;
				barycentric[i].x = a;
				barycentric[i].y = b;
				barycentric[i].z = c;
				barycentric[i].w = d;
				tempflag ++;//record how long the particle has been tracked under the RF field, in unit of dt;
			}

			else
			{
				if (a < -1e-15)
				{
					intersect(nodePosition[0], nodePosition[1], nodePosition[2], pold, pnew, &interpara[0]);
					intersec[0].x = pold.x + interpara[0] * (pnew.x - pold.x);
					intersec[0].y = pold.y + interpara[0] * (pnew.y - pold.y);
					intersec[0].z = pold.z + interpara[0] * (pnew.z - pold.z);
					alpha_beta(nodePosition[0], nodePosition[1], nodePosition[2], intersec[0], &ab[0]);
				}
				if (b < -1e-15)
				{
					intersect(nodePosition[2], nodePosition[3], nodePosition[0], pold, pnew, &interpara[1]);
					intersec[1].x = pold.x + interpara[1] * (pnew.x - pold.x);
					intersec[1].y = pold.y + interpara[1] * (pnew.y - pold.y);
					intersec[1].z = pold.z + interpara[1] * (pnew.z - pold.z);
					alpha_beta(nodePosition[2], nodePosition[3], nodePosition[0], intersec[1], &ab[1]);
				}
				if (c < -1e-15)
				{
					intersect(nodePosition[3], nodePosition[1], nodePosition[0], pold, pnew, &interpara[2]);
					intersec[2].x = pold.x + interpara[2] * (pnew.x - pold.x);
					intersec[2].y = pold.y + interpara[2] * (pnew.y - pold.y);
					intersec[2].z = pold.z + interpara[2] * (pnew.z - pold.z);
					alpha_beta(nodePosition[3], nodePosition[1], nodePosition[0], intersec[2], &ab[2]);
				}
				if (d < -1e-15)
				{
					intersect(nodePosition[1], nodePosition[3], nodePosition[2], pold, pnew, &interpara[3]);
					intersec[3].x = pold.x + interpara[3] * (pnew.x - pold.x);
					intersec[3].y = pold.y + interpara[3] * (pnew.y - pold.y);
					intersec[3].z = pold.z + interpara[3] * (pnew.z - pold.z);
					alpha_beta(nodePosition[1], nodePosition[3], nodePosition[2], intersec[3], &ab[3]);
				}
				for (j = 0; j < 4; j++)
				{
					if (ab[j].x <= 1 && ab[j].x >= -1e-15 && ab[j].y <= 1 && ab[j].y > -1e-15 && ab[j].x + ab[j].y <= 1)
					{
						if (oldmesh[tetindex*tetsize + j + 5] != -1)//if the surface the particle hit is not a shared surface
						{
							if (tempflag > -1)//if the particle didn't hit a wall from previous time step
							{
								found = 1;
								barycentric[i].x = a > -1e-15 ? a : 0;
								barycentric[i].y = b > -1e-15 ? b : 0;
								barycentric[i].z = c > -1e-15 ? c : 0;
								barycentric[i].w = d > -1e-15 ? d : 0;
								tempflag = -1;//register it as "just hit a wall"

								D_impactenergy[i*N_cycles * 2 + impact[i]] = 
									Vc*(sqrt(d_momentumf[i].x*d_momentumf[i].x + d_momentumf[i].y*d_momentumf[i].y + d_momentumf[i].z*d_momentumf[i].z + me*me*Vc*Vc)-me*Vc)/qe;
								impact[i] ++;

								pnew.x = pold.x + interpara[j] * 0.5 * (pnew.x - pold.x);
								pnew.y = pold.y + interpara[j] * 0.5 * (pnew.y - pold.y);
								pnew.z = pold.z + interpara[j] * 0.5 * (pnew.z - pold.z);

								d_momentumf[i].x = 0;
								d_momentumf[i].y = 0;
								d_momentumf[i].z = 0;
								momentum[i].x = 0;
								momentum[i].y = 0;
								momentum[i].z = 0;
								j = 5;
							}
							else//if the particle hit a wall in last time step
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
								d_momentumf[i].x = 0;
								d_momentumf[i].y = 0;
								d_momentumf[i].z = 0;
								momentum[i].x = 0;
								momentum[i].y = 0;
								momentum[i].z = 0;
								j = 5;
							}
						}
						else
						{
							tetindex = oldmesh[tetindex*tetsize + j + 10];//if particle hits a shared wall, then it moves to the neighbor tet
							j = 5;//jump out the for loop and start to search for tet again
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

				d_momentumf[i].x = 0;
				d_momentumf[i].y = 0;
				d_momentumf[i].z = 0;
				momentum[i].x = 0;
				momentum[i].y = 0;
				momentum[i].z = 0;
			}
		}
	}

	positionold[i].x = pnew.x / norm;
	positionold[i].y = pnew.y / norm;
	positionold[i].z = pnew.z / norm;
	positionnew[i].x = pnew.x ;
	positionnew[i].y = pnew.y ;
	positionnew[i].z = pnew.z ;
	momentum[i].x = d_momentumf[i].x;
	momentum[i].y = d_momentumf[i].y;
	momentum[i].z = d_momentumf[i].z;
	meshindextet[i] = tetindex;
	flag[i] = tempflag;
	
}

/*Update the momentum*/
__device__ double3 p_function(double3 momentum, double3 Efield, double3 Bfield, double t, double fre, double phi0)
{
	double3 dpodt;
	double qe = 1.6e-19;
	double Vc = 299792458.0;
	double me = 9.10938291e-31;
	double E = sqrt(momentum.x*momentum.x + momentum.y*momentum.y + momentum.z*momentum.z + me*me*Vc*Vc);
	double Phi = 2.0*M_PI*fre*t + phi0;
	double fsin, fcos;
	sincos(Phi, &fsin, &fcos);
	Efield.x = Efield.x*fsin;
	Efield.y = Efield.y*fsin;
	Efield.z = Efield.z*fsin;
	Bfield.x = -Bfield.x*fcos;
	Bfield.y = -Bfield.y*fcos;
	Bfield.z = -Bfield.z*fcos;

	dpodt.x = -qe*(Efield.x + Vc*(momentum.y*Bfield.z - momentum.z*Bfield.y)
		/ E);
	dpodt.y = -qe*(Efield.y + Vc*(-momentum.x*Bfield.z + momentum.z*Bfield.x)
		/ E);
	dpodt.z = -qe*(Efield.z + Vc*(momentum.x*Bfield.y - momentum.y*Bfield.x)
		/ E);
	return dpodt;
}
__global__ void getField(double4* D_barycentric, double3* D_p_Efd, double3* D_p_Bfd,  double3* Efield, double3* Bfield, double fdnorm,
	int* D_meshindextet, int* D_tetmesh, int N_par, int tetsize)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;

	double3 dfdtemp[4];
	double4 barycentrictemp;
	int j;
	int tetindex = D_meshindextet[i];
	if (i < N_par)
	{
		barycentrictemp.x = D_barycentric[i].x;
		barycentrictemp.y = D_barycentric[i].y;
		barycentrictemp.z = D_barycentric[i].z;
		barycentrictemp.w = D_barycentric[i].w;

		/*get the field for four vertex of the tet where the particle was found*/
		for (j = 0; j < 4; j++)
		{
			dfdtemp[j].x = Efield[D_tetmesh[tetindex*tetsize + j + 1]].x * fdnorm;
			dfdtemp[j].y = Efield[D_tetmesh[tetindex*tetsize + j + 1]].y * fdnorm;
			dfdtemp[j].z = Efield[D_tetmesh[tetindex*tetsize + j + 1]].z * fdnorm;
		}

		D_p_Efd[i].x = barycentrictemp.x *dfdtemp[3].x + barycentrictemp.y*dfdtemp[1].x + barycentrictemp.z*dfdtemp[2].x + barycentrictemp.w *dfdtemp[0].x;
		D_p_Efd[i].y = barycentrictemp.x *dfdtemp[3].y + barycentrictemp.y*dfdtemp[1].y + barycentrictemp.z*dfdtemp[2].y + barycentrictemp.w *dfdtemp[0].y;
		D_p_Efd[i].z = barycentrictemp.x *dfdtemp[3].z + barycentrictemp.y*dfdtemp[1].z + barycentrictemp.z*dfdtemp[2].z + barycentrictemp.w *dfdtemp[0].z;
		for (j = 0; j < 4; j++)
		{
			dfdtemp[j].x = Bfield[D_tetmesh[tetindex*tetsize + j + 1]].x * fdnorm;
			dfdtemp[j].y = Bfield[D_tetmesh[tetindex*tetsize + j + 1]].y * fdnorm;
			dfdtemp[j].z = Bfield[D_tetmesh[tetindex*tetsize + j + 1]].z * fdnorm;
		}
		D_p_Bfd[i].x = barycentrictemp.x *dfdtemp[3].x + barycentrictemp.y*dfdtemp[1].x + barycentrictemp.z*dfdtemp[2].x + barycentrictemp.w *dfdtemp[0].x;
		D_p_Bfd[i].y = barycentrictemp.x *dfdtemp[3].y + barycentrictemp.y*dfdtemp[1].y + barycentrictemp.z*dfdtemp[2].y + barycentrictemp.w *dfdtemp[0].y;
		D_p_Bfd[i].z = barycentrictemp.x *dfdtemp[3].z + barycentrictemp.y*dfdtemp[1].z + barycentrictemp.z*dfdtemp[2].z + barycentrictemp.w *dfdtemp[0].z;
	}
	
}
__global__ void rungeKutta(double3* D_positionold, double3* D_position0, double3* D_momentum0, double3* D_p_Efd, double3* D_p_Bfd,
	double t, double dt, double fre, double norm,
	int* D_initphase, int* flag,
	int N_par, int phase_step)
{
	t = t / fre;
	dt = dt / fre;
	double qe = 1.6e-19;
	double Vc = 299792458.0;
	double me = 9.10938291e-31;
	double E;
	double Phi;
	double fsin, fcos;
	int tempflag;
	double3 Efield, Bfield, Efield0, Bfield0;
	double3 dpodt;
	double3 momentum0, momentumt, momentumf;
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < N_par)
	{
		tempflag = flag[i];
		if (tempflag >= -1)
		{
			momentum0.x = D_momentum0[i].x;
			momentum0.y = D_momentum0[i].y;
			momentum0.z = D_momentum0[i].z;
			momentumf.x = momentum0.x;
			momentumf.y = momentum0.y;
			momentumf.z = momentum0.z;
			momentumt.x = momentum0.x;
			momentumt.y = momentum0.y;
			momentumt.z = momentum0.z;
			Efield0.x = D_p_Efd[i].x;
			Efield0.y = D_p_Efd[i].y;
			Efield0.z = D_p_Efd[i].z;
			Bfield0.x = -D_p_Bfd[i].x;
			Bfield0.y = -D_p_Bfd[i].y;
			Bfield0.z = -D_p_Bfd[i].z;
			/*First*/
			Phi = 2.0*M_PI*fre*t + (double)(D_initphase[i]) / (double)phase_step*M_PI;
			sincos(Phi, &fsin, &fcos);
			Efield.x = Efield0.x*fsin;
			Efield.y = Efield0.y*fsin;
			Efield.z = Efield0.z*fsin;
			Bfield.x = Bfield0.x*fcos;
			Bfield.y = Bfield0.y*fcos;
			Bfield.z = Bfield0.z*fcos;
			E = sqrt(momentumt.x*momentumt.x + momentumt.y*momentumt.y + momentumt.z*momentumt.z + me*me*Vc*Vc);
			dpodt.x = -qe*(Efield.x + Vc*(momentumt.y*Bfield.z - momentumt.z*Bfield.y)
				/ E);
			dpodt.y = -qe*(Efield.y + Vc*(-momentumt.x*Bfield.z + momentumt.z*Bfield.x)
				/ E);
			dpodt.z = -qe*(Efield.z + Vc*(momentumt.x*Bfield.y - momentumt.y*Bfield.x)
				/ E);
			momentumt.x = momentum0.x + 0.5 * dpodt.x*dt;
			momentumt.y = momentum0.y + 0.5 * dpodt.y*dt;
			momentumt.z = momentum0.z + 0.5 * dpodt.z*dt;
			momentumf.x += dpodt.x*dt / 6.0;
			momentumf.y += dpodt.y*dt / 6.0;
			momentumf.z += dpodt.z*dt / 6.0;
			///*Second*/
			//Phi = 2.0*M_PI*fre*(t + dt / 2.0) + (double)(D_initphase[i]) / (double)phase_step*M_PI;
			//sincos(Phi, &fsin, &fcos);
			//Efield.x = Efield0.x*fsin;
			//Efield.y = Efield0.y*fsin;
			//Efield.z = Efield0.z*fsin;
			//Bfield.x = Bfield0.x*fcos;
			//Bfield.y = Bfield0.y*fcos;
			//Bfield.z = Bfield0.z*fcos;
			//E = sqrt(momentumt.x*momentumt.x + momentumt.y*momentumt.y + momentumt.z*momentumt.z + me*me*Vc*Vc);
			//dpodt.x = -qe*(Efield.x + Vc*(momentumt.y*Bfield.z - momentumt.z*Bfield.y)
			//	/ E);
			//dpodt.y = -qe*(Efield.y + Vc*(-momentumt.x*Bfield.z + momentumt.z*Bfield.x)
			//	/ E);
			//dpodt.z = -qe*(Efield.z + Vc*(momentumt.x*Bfield.y - momentumt.y*Bfield.x)
			//	/ E);
			//momentumt.x = momentum0.x + 0.5 * dpodt.x*dt;
			//momentumt.y = momentum0.y + 0.5 * dpodt.y*dt;
			//momentumt.z = momentum0.z + 0.5 * dpodt.z*dt;
			//momentumf.x += dpodt.x*dt / 3.0;
			//momentumf.y += dpodt.y*dt / 3.0;
			//momentumf.z += dpodt.z*dt / 3.0;
			///*Third*/
			//Phi = 2.0*M_PI*fre*(t + dt / 2.0) + (double)(D_initphase[i]) / (double)phase_step*M_PI;
			//sincos(Phi, &fsin, &fcos);

			//Efield.x = Efield0.x*fsin;
			//Efield.y = Efield0.y*fsin;
			//Efield.z = Efield0.z*fsin;
			//Bfield.x = Bfield0.x*fcos;
			//Bfield.y = Bfield0.y*fcos;
			//Bfield.z = Bfield0.z*fcos;
			//E = sqrt(momentumt.x*momentumt.x + momentumt.y*momentumt.y + momentumt.z*momentumt.z + me*me*Vc*Vc);
			//dpodt.x = -qe*(Efield.x + Vc*(momentumt.y*Bfield.z - momentumt.z*Bfield.y)
			//	/ E);
			//dpodt.y = -qe*(Efield.y + Vc*(-momentumt.x*Bfield.z + momentumt.z*Bfield.x)
			//	/ E);
			//dpodt.z = -qe*(Efield.z + Vc*(momentumt.x*Bfield.y - momentumt.y*Bfield.x)
			//	/ E);
			//momentumt.x = momentum0.x + dpodt.x*dt;
			//momentumt.y = momentum0.y + dpodt.y*dt;
			//momentumt.z = momentum0.z + dpodt.z*dt;
			//momentumf.x += dpodt.x*dt / 3.0;
			//momentumf.y += dpodt.y*dt / 3.0;
			//momentumf.z += dpodt.z*dt / 3.0;
			///*Fourth*/
			//Phi = 2.0*M_PI*fre*(t + dt) + (double)(D_initphase[i]) / (double)phase_step*M_PI;
			//sincos(Phi, &fsin, &fcos);

			//Efield.x = Efield0.x*fsin;
			//Efield.y = Efield0.y*fsin;
			//Efield.z = Efield0.z*fsin;
			//Bfield.x = Bfield0.x*fcos;
			//Bfield.y = Bfield0.y*fcos;
			//Bfield.z = Bfield0.z*fcos;
			//E = sqrt(momentumt.x*momentumt.x + momentumt.y*momentumt.y + momentumt.z*momentumt.z + me*me*Vc*Vc);
			//dpodt.x = -qe*(Efield.x + Vc*(momentumt.y*Bfield.z - momentumt.z*Bfield.y)
			//	/ E);
			//dpodt.y = -qe*(Efield.y + Vc*(-momentumt.x*Bfield.z + momentumt.z*Bfield.x)
			//	/ E);
			//dpodt.z = -qe*(Efield.z + Vc*(momentumt.x*Bfield.y - momentumt.y*Bfield.x)
			//	/ E);
			//momentumf.x += dpodt.x*dt / 6.0;
			//momentumf.y += dpodt.y*dt / 6.0;
			//momentumf.z += dpodt.z*dt / 6.0;

			//E = sqrt(momentumf.x*momentumf.x + momentumf.y*momentumf.y + momentumf.z*momentumf.z + me*me*Vc*Vc);

			D_momentum0[i].x = momentumf.x;
			D_momentum0[i].y = momentumf.y;
			D_momentum0[i].z = momentumf.z;

			/*D_position0[i].x = D_positionold[i].x*norm + momentumf.x / E*Vc*dt;
			D_position0[i].y = D_positionold[i].y*norm + momentumf.y / E*Vc*dt;
			D_position0[i].z = D_positionold[i].z*norm + momentumf.z / E*Vc*dt;*/
		}
	}
}
/*update the position of partiles*/
__global__ void moveKernel(double3 *position, double3 *momentum, double3* d_position0, double3* d_momentumf, double3* D_p_Efd, double3* D_p_Bfd, 
	double norm, double t, double dt, double fre, 
	int* initphase, int* flag, 
	int N_par, int phase_step)
{
	
	double Vc = 299792458.0;
	double me = 9.10938291e-31;
	double phi0;

	double3 pEfd, pBfd;//fields at particle location
	double3 position0;
	double3 momentumt, momentum0, momentum1, momentum2, momentum3, momentum4, momentumf;
	double3 v;


	t = t / fre;
	dt = dt / fre;
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j;
	int tempflag;
	if (i < N_par )
	{
		tempflag = flag[i];
		if (tempflag >= -1)
		{
			position0.x = position[i].x*norm;
			position0.y = position[i].y*norm;
			position0.z = position[i].z*norm;
			momentum0.x = momentum[i].x;
			momentum0.y = momentum[i].y;
			momentum0.z = momentum[i].z;
			
			pEfd.x = D_p_Efd[i].x;
			pEfd.y = D_p_Efd[i].y;
			pEfd.z = D_p_Efd[i].z;
			pBfd.x = D_p_Bfd[i].x;
			pBfd.y = D_p_Bfd[i].y;
			pBfd.z = D_p_Bfd[i].z;

			phi0 = (double)(initphase[i]) / phase_step*M_PI;

			//Runge-Kutta method to find the next momentum.
			momentumf = p_function(momentum0, pEfd, pBfd, t, fre, phi0);
			momentum1.x = momentumf.x*dt;
			momentum1.y = momentumf.y*dt;
			momentum1.z = momentumf.z*dt;

			momentumt.x = momentum0.x + momentum1.x / 2.0;
			momentumt.y = momentum0.y + momentum1.y / 2.0;
			momentumt.z = momentum0.z + momentum1.z / 2.0;
			momentumf = p_function(momentumt, pEfd, pBfd, t + dt / 2.0, fre, phi0);
			momentum2.x = momentumf.x*dt;
			momentum2.y = momentumf.y*dt;
			momentum2.z = momentumf.z*dt;

			momentumt.x = momentum0.x + momentum2.x / 2.0;
			momentumt.y = momentum0.y + momentum2.y / 2.0;
			momentumt.z = momentum0.z + momentum2.z / 2.0;
			momentumf = p_function(momentumt, pEfd, pBfd, t + dt / 2.0, fre, phi0);
			momentum3.x = momentumf.x*dt;
			momentum3.y = momentumf.y*dt;
			momentum3.z = momentumf.z*dt;

			momentumt.x = momentum0.x + momentum3.x;
			momentumt.y = momentum0.y + momentum3.y;
			momentumt.z = momentum0.z + momentum3.z;
			momentumf = p_function(momentumt, pEfd, pBfd, t + dt, fre, phi0);
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
			d_position0[i].x = position0.x;
			d_position0[i].y = position0.y;
			d_position0[i].z = position0.z;

			d_momentumf[i].x = momentumf.x;
			d_momentumf[i].y = momentumf.y;
			d_momentumf[i].z = momentumf.z;

		}

	}
}
__global__ void dumpimpat(double3* impactengergy, int N_par);


void activetet(double* nodes, double* xrange, double* yrange, double* zrange, double norm, int* tetall, int* meshindextet_temp, int* N_par, int nall,int tetsize)
{
	/*Find the tet that is in the given box*/
	int i, j, k;
	double3 *centers;
	std::vector<int> tempactive;
	centers = new double3[nall];
	for (i = 0; i < nall; i++)
	{
		centers[i].x = (nodes[tetall[i*tetsize + 1] * 3] + nodes[tetall[i*tetsize + 2] * 3] + nodes[tetall[i*tetsize + 3] * 3] + nodes[tetall[i*tetsize + 4] * 3])/4;
		centers[i].y = (nodes[tetall[i*tetsize + 1] * 3 + 1] + nodes[tetall[i*tetsize + 2] * 3 + 1] + nodes[tetall[i*tetsize + 3] * 3 + 1] + nodes[tetall[i*tetsize + 4] * 3 + 1]) / 4 ;
		centers[i].z = (nodes[tetall[i*tetsize + 1] * 3 + 2] + nodes[tetall[i*tetsize + 2] * 3 + 2] + nodes[tetall[i*tetsize + 3] * 3 + 2] + nodes[tetall[i*tetsize + 4] * 3 + 2]) / 4 ;
	}
	for (i = 0; i < nall; i++)
	{
		if (centers[i].x > xrange[0] && centers[i].x<xrange[1] && centers[i].y>yrange[0] && centers[i].y<yrange[1] && centers[i].z>zrange[0] && centers[i].z < zrange[1]
			&& (tetall[i*tetsize + 5] != -1 || tetall[i*tetsize + 6] != -1 || tetall[i*tetsize + 7] != -1 || tetall[i*tetsize + 8] != -1))
		{
			tempactive.push_back(i);
		}
	}
	j = tempactive.size();
	for (i = 0; i < j; i++)
	{
		meshindextet_temp[i] = tempactive[i];
	}
	*N_par = j*phase_step;
	/*std::ofstream activefile("active.txt");
	for (i = 0; i < j; i++)
	{
		activefile << meshindextet_temp[i] << std::endl;
	}*/
	std::cout << "Activate tet finished, N_par = " <<*N_par<< std::endl;
}
void showcudaerror()
{
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cout << "CUDA Error:"<<cudaGetErrorString(err)<<std::endl;
	}
}
void getinput()
{
	std::string line;
	std::string item;
	std::ifstream myfile;
	std::vector<std::string> elems;
	
	char delim;
	char waitchar;
	delim = ':';

	myfile.open("input.txt");
	if (myfile.is_open())
	{
		while (std::getline(myfile, line))
		{
			std::stringstream ss(line);
			while (std::getline(ss, item, delim))
			{
				elems.push_back(item);
			}
			if (elems.size() > 0)
			{
				elems[0].erase(std::remove_if(elems[0].begin(), elems[0].end(), isspace), elems[0].end());
				elems[1].erase(std::remove_if(elems[1].begin(), elems[1].end(), isspace), elems[1].end());
				elems[1].erase(std::remove(elems[1].begin(), elems[1].end(), ';'), elems[1].end());
				inputs[elems[0]] = elems[1];
				elems.clear();
			}
		}
		myfile.close();
		FILE_NAME = inputs["ModelFile"].c_str();
		FILE_NAME2 = inputs["FieldFile"].c_str();
		isGPU = stoi(inputs["isGPU"]);
		fdnorm = stod(inputs["FieldNorm_min"]);
		fdnorm_max = stod(inputs["FieldNorm_max"]);
		fdnorm_step = stod(inputs["FieldNorm_step"]);
		xrange[0] = stod(inputs["X_min"]);
		xrange[1] = stod(inputs["X_max"]);
		yrange[0] = stod(inputs["Y_min"]);
		yrange[1] = stod(inputs["Y_max"]);
		zrange[0] = stod(inputs["Z_min"]);
		zrange[1] = stod(inputs["Z_max"]);
		dt = stod(inputs["dt"]);
		phase_step = stod(inputs["Phase_sample_steps"]);
		N_cycles = stod(inputs["N_cyclse"]);
		initenergy = stod(inputs["Initial_Energy"]);
	}
	else
	{
		std::cout << "Couldn't find input.txt" << std::endl;
		exit(4);
	}

	
}
void shuffle(double4* barycentric, double3* position, double3* momentum, double* impactenergy, int* flag, int* meshindextet, int* impact,int* intiphase,
	double4* barycentric_shuffle, double3* position_shuffle, double3* momentum_shuffle, double* impactenergy_shuffle, int* flag_shuffle, int* meshindextet_shuffle, int* impact_shuffle, int* initphase_shuffle,int* N_par)
{
	int i,j;
	int countLive = 0;
	for (i = 0; i < *N_par; i++)
	{
		if ((flag[i]>-2)&&flag[i]<10/dt)//not dead due to lost or hitting the wall, and not slowly flying in the field for too long (>10 T)
		{
			barycentric_shuffle[countLive] = barycentric[i];
			position_shuffle[countLive] = position[i];
			momentum_shuffle[countLive] = momentum[i];
			flag_shuffle[countLive] = flag[i];
			meshindextet_shuffle[countLive] = meshindextet[i];
			impact_shuffle[countLive] = impact[i];
			initphase_shuffle[countLive] = initphase[i];
			memcpy((void*)(impactenergy_shuffle+countLive*N_cycles * 2), (void*)(impactenergy+i*N_cycles * 2), N_cycles * 2*sizeof(double));
			countLive++;
		}
	}
	*N_par = countLive;
}
void launch_kernel(double3 *pos, double3 *vel, double frametime, int func)
{
	int i,j;
	// execute the kernel
	dim3 block(256, 1, 1);
	dim3 grid(N_par/block.x+1,1,1);
	cudaError_t err;
	if (func == 0)
	{
		int* dindex;//indexes for exterior triangles for device
		cudaMalloc(&dindex, ntriangles*sizeof(int)*3);
		cudaMemcpy(dindex, indexes, ntriangles * sizeof(int)*3, cudaMemcpyHostToDevice);

		cudaMalloc(&dnodes, nnodes*sizeof(double)*3);
		cudaMemcpy(dnodes, nodes, nnodes * sizeof(double)*3, cudaMemcpyHostToDevice);
		
		//initialize the barycentric coordinates of the particles to zeros;
		
		for (i = 0; i < N_par; i++)
		{
			barycentric[i].w = 0;
			barycentric[i].x = 0;
			barycentric[i].y = 0;
			barycentric[i].z = 0;
		}
		
		cudaMemcpy(dbarycentric, barycentric, N_par*sizeof(double4), cudaMemcpyHostToDevice);
		//activetet(nodes, xrange, yrange, zrange, norm, tetall, meshindextet, &N_par, nall, tetsize);
		for (i = 0; i < N_par / phase_step; i++)//change the tet index every phase_step numbers in meshindextet 
		{
			for (j = 0; j < phase_step; j++)
			{
				meshindextet[i*phase_step + j] = meshindextet_temp[i];
			}
		}
		cudaMemcpy(dmeshindextet, meshindextet, N_par*sizeof(int), cudaMemcpyHostToDevice);
		/*initialize the particle positions and momentum*/
		initpar << <grid, block >> >(dbarycentric, pos, vel, d_position0, d_momentumf, D_momentumt, D_impactenergy, dnodes, dvolume, norm, dimpact, dmeshindextet, dflag, dtetall, dinitphase, N_par, N_cycles, tetsize, phase_step);
		
		if (isGPU == 0)
		{
			cudaMemcpy(barycentric, dbarycentric, N_par * sizeof(double4), cudaMemcpyDeviceToHost);
			cudaMemcpy(Efd_temp, dEfd, nnodes * sizeof(double3), cudaMemcpyDeviceToHost);
			cudaMemcpy(Bfd_temp, dBfd, nnodes * sizeof(double3), cudaMemcpyDeviceToHost);
			cudaMemcpy(meshindextet, dmeshindextet, sizeof(int)*N_par, cudaMemcpyDeviceToHost);
			cudaMemcpy(impact, dimpact, sizeof(int)*N_par, cudaMemcpyDeviceToHost);
			cudaMemcpy(impactenergy, D_impactenergy, sizeof(double)*N_par*N_cycles * 2, cudaMemcpyDeviceToHost);
			cudaMemcpy(flag, dflag, sizeof(int)*N_par, cudaMemcpyDeviceToHost);
			cudaMemcpy(Hposition, pos, sizeof(double3)*N_par, cudaMemcpyDeviceToHost);
			cudaMemcpy(Hvelocity, vel, sizeof(double3)*N_par, cudaMemcpyDeviceToHost);
			cudaMemcpy(H_position0, d_position0, sizeof(double3)*N_par, cudaMemcpyDeviceToHost);
			cudaMemcpy(H_momentumf, d_momentumf, sizeof(double3)*N_par, cudaMemcpyDeviceToHost);
			cudaMemcpy(initphase, dinitphase, sizeof(int)*N_par, cudaMemcpyDeviceToHost);
		}
		cudaMemcpy(barycentric, dbarycentric, N_par * sizeof(double4), cudaMemcpyDeviceToHost);
		cudaMemcpy(Efd_temp, dEfd, nnodes * sizeof(double3), cudaMemcpyDeviceToHost);
		cudaMemcpy(Bfd_temp, dBfd, nnodes * sizeof(double3), cudaMemcpyDeviceToHost);
		cudaMemcpy(meshindextet, dmeshindextet, sizeof(int)*N_par, cudaMemcpyDeviceToHost);
		cudaMemcpy(impact, dimpact, sizeof(int)*N_par, cudaMemcpyDeviceToHost);
		cudaMemcpy(impactenergy, D_impactenergy, sizeof(double)*N_par*N_cycles * 2, cudaMemcpyDeviceToHost);
		cudaMemcpy(flag, dflag, sizeof(int)*N_par, cudaMemcpyDeviceToHost);
		cudaMemcpy(Hposition, pos, sizeof(double3)*N_par, cudaMemcpyDeviceToHost);
		cudaMemcpy(Hvelocity, vel, sizeof(double3)*N_par, cudaMemcpyDeviceToHost);
		cudaMemcpy(H_position0, d_position0, sizeof(double3)*N_par, cudaMemcpyDeviceToHost);
		cudaMemcpy(H_momentumf, d_momentumf, sizeof(double3)*N_par, cudaMemcpyDeviceToHost);
		cudaMemcpy(initphase, dinitphase, sizeof(int)*N_par, cudaMemcpyDeviceToHost);
		std::cout << "Trying to initialize particles" << std::endl;
		printf("Field Normalizer = %2.3e  \n", fdnorm);
		printf("Initial Phase = %2.2f \n", g_fAnim);
		printf("dt = %2.3f*T \n", dt);

	}
	else
	{
		if (isGPU==1)
		{
			if (frametime>2&&(int)(frametime/dt)%(int)(2/dt) == 0)
			{
				cudaMemcpy(Hposition, pos, sizeof(double3)*N_par, cudaMemcpyDeviceToHost);
				cudaMemcpy(Hvelocity, vel, sizeof(double3)*N_par, cudaMemcpyDeviceToHost);
				cudaMemcpy(flag, dflag, sizeof(int)*N_par, cudaMemcpyDeviceToHost);
				cudaMemcpy(impact, dimpact, sizeof(int)*N_par, cudaMemcpyDeviceToHost);
				cudaMemcpy(meshindextet, dmeshindextet, sizeof(int)*N_par, cudaMemcpyDeviceToHost);
				cudaMemcpy(barycentric, dbarycentric, sizeof(double4)*N_par, cudaMemcpyDeviceToHost);
				cudaMemcpy(impactenergy, D_impactenergy, sizeof(double)*N_par*N_cycles*2, cudaMemcpyDeviceToHost);
				int countLive;//count the number of live particles
				int i;
				for (i = 0; i < N_par; i++)
				{
					countLive += (flag[i]>-2&&flag[i]<5/dt) ? 1 : 0;//if the flag of the particle is larger than -2, we count it as live.
				}
				/*get rid of the dead particles*/
				shuffle(barycentric, Hposition, Hvelocity, impactenergy, flag, meshindextet, impact,initphase,
					barycentric_shuffle, Hposition_shuffle, Hvelocity_shuffle, impactenergy_shuffle, flag_shuffle, meshindextet_shuffle, impact_shuffle, initphase_shuffle,&N_par);

				memcpy(barycentric, barycentric_shuffle, sizeof(double4)*N_par);
				memcpy(Hposition, Hposition_shuffle, sizeof(double3)*N_par);
				memcpy(Hvelocity, Hvelocity_shuffle, sizeof(double3)*N_par);
				memcpy(flag, flag_shuffle, sizeof(int)*N_par);
				memcpy(meshindextet, meshindextet_shuffle, sizeof(int)*N_par);
				memcpy(impact, impact_shuffle, sizeof(int)*N_par);
				memcpy(initphase, initphase_shuffle, sizeof(int)*N_par);
				memcpy(impactenergy, impactenergy_shuffle, sizeof(double)*N_par*N_cycles*2);
/*
				barycentric = barycentric_shuffle;
				Hposition = Hposition_shuffle;
				Hvelocity = Hvelocity_shuffle;
				flag = flag_shuffle;
				meshindextet = meshindextet_shuffle;
				impact = impact_shuffle;
				impactenergy = impactenergy_shuffle;*/

				cudaMemcpy(pos, Hposition, sizeof(double3)*N_par, cudaMemcpyHostToDevice);
				cudaMemcpy(vel, Hvelocity, sizeof(double3)*N_par, cudaMemcpyHostToDevice);
				cudaMemcpy(dflag, flag, sizeof(int)*N_par, cudaMemcpyHostToDevice);
				cudaMemcpy(dimpact, impact, sizeof(int)*N_par, cudaMemcpyHostToDevice);
				cudaMemcpy(dinitphase, initphase, sizeof(int)*N_par, cudaMemcpyHostToDevice);
				cudaMemcpy(dmeshindextet, meshindextet, sizeof(int)*N_par, cudaMemcpyHostToDevice);
				cudaMemcpy(dbarycentric, barycentric, sizeof(double4)*N_par, cudaMemcpyHostToDevice);
				cudaMemcpy(D_impactenergy, impactenergy, sizeof(double)*N_par*N_cycles * 2, cudaMemcpyHostToDevice);

			}

			/*Move the particle by one time step*/
			block.x = 128;
			grid.x = N_par / block.x + 1;
			getField <<< grid, block >>> (dbarycentric, D_p_Efd, D_p_Bfd, dEfd, dBfd, fdnorm,
				dmeshindextet, dtetall, N_par,tetsize);
			block.x = 64;
			grid.x = N_par / block.x + 1;
			//rungeKutta << < grid, block >> > (pos, d_position0, d_momentumf, D_p_Efd, D_p_Bfd, frametime, dt, fre, norm, dinitphase, dflag,N_par, phase_step);
			moveKernel << < grid, block >> >(pos, vel, d_position0, d_momentumf, D_p_Efd, D_p_Bfd, 
				norm, frametime, dt, fre, dinitphase, dflag, N_par, phase_step);
			/*Locate the particle*/
			block.x = 128;
			grid.x = N_par / block.x + 1;
			tracking << <grid, block >> >(dbarycentric, pos, d_position0, vel, d_momentumf, D_impactenergy, norm, dnodes, dvolume, dmeshindextet, dtetall, dflag, dimpact, N_par, N_cycles, tetsize);
			cudaMemcpy(Hposition, pos, sizeof(double3)*N_par, cudaMemcpyDeviceToHost);
			cudaMemcpy(flag, dflag, sizeof(int)*N_par, cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
		}
		
		else
		{
			timer0=clock();
			moveKernelHost(barycentric, Hposition, Hvelocity, H_position0, H_momentumf,Efd_temp, Bfd_temp, impactmomentum, 
				nodes, volume, 
				norm, frametime,dt, fre, fdnorm, 
				impact, meshindextet, tetall, flag, initphase,
				tetsize,N_par, N_cycles, phase_step);
			std::cout << "Time to run MoveKernel on CPU is " << ((double)clock() - (double)timer0) / CLOCKS_PER_SEC*1000.0 << " mS\n";

			timer0 = clock();
			trackingHost(barycentric, Hposition, H_position0, Hvelocity, H_momentumf, impactmomentum, norm, nodes, volume, meshindextet, tetall, flag, impact, N_par,
				N_cycles, tetsize);
			std::cout << "Time to run tracking on CPU is " << ((double)clock() - (double)timer0) / CLOCKS_PER_SEC*1000.0 << " mS\n";
			cudaMemcpy(pos, Hposition, N_par * sizeof(double3), cudaMemcpyHostToDevice);
			cudaMemcpy(vel, Hvelocity, N_par * sizeof(double3), cudaMemcpyHostToDevice);
		}
		

	}
}

bool checkHW(char *name, const char *gpuType, int dev)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	strcpy(name, deviceProp.name);

	if (!STRNCASECMP(deviceProp.name, gpuType, strlen(gpuType)))
	{
		return true;
	}
	else
	{
		return false;
	}
}

int findGraphicsGPU(char *name)
{
	int nGraphicsGPU = 0;
	int deviceCount = 0;
	bool bFoundGraphics = false;
	char firstGraphicsName[256], temp[256];

	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess)
	{
		printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		printf("> FAILED %s sample finished, exiting...\n", sSDKsample);
		exit(EXIT_FAILURE);
	}

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0)
	{
		printf("> There are no device(s) supporting CUDA\n");
		return false;
	}
	else
	{
		printf("> Found %d CUDA Capable Device(s)\n", deviceCount);
	}

	for (int dev = 0; dev < deviceCount; ++dev)
	{
		bool bGraphics = !checkHW(temp, (const char *)"Tesla", dev);
		printf("> %s\t\tGPU %d: %s\n", (bGraphics ? "Graphics" : "Compute"), dev, temp);

		if (bGraphics)
		{
			if (!bFoundGraphics)
			{
				strcpy(firstGraphicsName, temp);
			}

			nGraphicsGPU++;
		}
	}

	if (nGraphicsGPU)
	{
		strcpy(name, firstGraphicsName);
	}
	else
	{
		strcpy(name, "this hardware");
	}

	return nGraphicsGPU;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{

	// Get the number of processors in this system
	int iCPU = omp_get_num_procs();
	std::cout << iCPU << std::endl;
	// Now set the number of threads
	omp_set_num_threads(iCPU);
	char *ref_file = NULL;
	getinput();
	std::string exitchar;
	/* Loop indexes, and error handling. */
	int i = 0, j = 0, k =0, l = 0;
	/* Open the file. NC_NOWRITE tells netCDF we want read-only access
	* to the file.*/
	int ncid;
	if ((retval = nc_open(FILE_NAME, NC_NOWRITE, &ncid)))
	{
		ERR(retval);
		std::cout << std::endl;
		std::cout << "Press Enter to Exit";
		std::getline(std::cin, exitchar);
		exit(ERRCODE);
	}
	printf("The File ID is: %d\n", ncid);
	int tempid1, tempid2;//temporay variables to store id info of interested dimensions
	size_t templ1, templ2;//temporary variables to store dimension length info.

	/*read the nodes info*/
	std::cout << "Start to read the nodes info from file." << std::endl;
	if ((retval = nc_inq_dimid(ncid, "ncoords", &tempid1)))//get the id of the dimension which name is ncoords.(which represent the number of nodes)
		ERR(retval);
	if ((retval = nc_inq_dimlen(ncid, tempid1, &templ1)))//get the length of the dimension that represent the number of nodes.
		ERR(retval);
	nnodes = templ1;//total length of the coordinates 
	nodes = new double[nnodes * 3];//construct the array for the nodes.
	nodesdisp = new double[nnodes * 3];
	if ((retval = nc_inq_varid(ncid, "coords", &tempid1)))//get the id of the variable that represent the nodes.
		ERR(retval);
	if ((retval = nc_get_var_double(ncid, tempid1, nodes)))//read in the value of node coordinates
		ERR(retval);

	mincor = std::min_element(nodes, nodes + templ1 - 1);
	maxcor = std::max_element(nodes, nodes + templ1 - 1);
	norm = abs(*mincor) >= abs(*maxcor) ? abs(*mincor) : abs(*maxcor);

	for (i = 0; i < nnodes * 3; i++)
	{
		nodesdisp[i] = nodes[i] / norm;
	}

	std::cout << "Nodes info readin complete." << std::endl;
	std::cout << "Writing the nodes info to a file" << std::endl;
	/*write the nodes to a file*/
	std::ofstream nodefile("nodes.txt");
	for (i = 0; i < nnodes; i++)
	{
		nodefile << "ID: "<<i<<"  "<<nodes[i * 3] << " , " << nodes[i * 3 + 1] << " , " << nodes[i * 3 + 2] << std::endl;
	}
	nodefile.close();
	std::cout << "Nodes info writing complete!" << std::endl;

	/*read the extiror tet info*/
	std::cout << "Start to read the exterior tet info from file." << std::endl;
	if ((retval = nc_inq_dimid(ncid, "tetexterior", &tempid1)))//get the id of the dimension which name is tetinterior.(which represent the number of interior tetrahedron)
		ERR(retval);
	if ((retval = nc_inq_dimlen(ncid, tempid1, &templ1)))//get the length of the dimension that represent the interior tet.
		ERR(retval);
	next = templ1;
	tetext = new int[next * 9];
	if ((retval = nc_inq_varid(ncid, "tetrahedron_exterior", &tempid1)))//get the id of the variable that represent the exterior tetrahedron.
		ERR(retval);
	if ((retval = nc_get_var_int(ncid, tempid1, tetext)))
		ERR(retval);

	for (i = 0; i < next; i++)
	{
		for (j = 0; j < 4; j++)
		{
			if (tetext[i * 9 + j + 5] != -1)//-1 means a shared surface with other tet
			{
				ntriangles++;
			}
		}
	}
	std::cout << "Exterior Tet info readin complete." << std::endl;

	/*Make the index array for the drawing*/
	std::cout << "Constructing the index array for model drawing" << std::endl;
	indexes = new int[ntriangles * 3];
	indexesedge = new int[ntriangles * 6];
	int tempi = 0;//temporary index;
	int tempiedge = 0;
	for (i = 0; i < next; i++)
	{
		for (j = 0; j < 4; j++)
		{
			if (tetext[i * 9 + j + 5] != -1)//-1 means a shared surface with other tet
			{
				switch (j)
				{
				case 0:
					indexes[tempi] = tetext[i * 9 + 1];
					indexesedge[tempiedge] = tetext[i * 9 + 1];
					tempiedge++;
					indexesedge[tempiedge] = tetext[i * 9 + 2];
					tempiedge++;
					tempi++;
					indexes[tempi] = tetext[i * 9 + 2];
					indexesedge[tempiedge] = tetext[i * 9 + 2];
					tempiedge++;
					indexesedge[tempiedge] = tetext[i * 9 + 3];
					tempiedge++;
					tempi++;
					indexes[tempi] = tetext[i * 9 + 3];
					indexesedge[tempiedge] = tetext[i * 9 + 3];
					tempiedge++;
					indexesedge[tempiedge] = tetext[i * 9 + 1];
					tempiedge++;
					tempi++;
					break;

				case 1:
					indexes[tempi] = tetext[i * 9 + 1];
					indexesedge[tempiedge] = tetext[i * 9 + 1];
					tempiedge++;
					indexesedge[tempiedge] = tetext[i * 9 + 3];
					tempiedge++;
					tempi++;
					indexes[tempi] = tetext[i * 9 + 3];
					indexesedge[tempiedge] = tetext[i * 9 + 3];
					tempiedge++;
					indexesedge[tempiedge] = tetext[i * 9 + 4];
					tempiedge++;
					tempi++;
					indexes[tempi] = tetext[i * 9 + 4];
					indexesedge[tempiedge] = tetext[i * 9 + 4];
					tempiedge++;
					indexesedge[tempiedge] = tetext[i * 9 + 1];
					tempiedge++;
					tempi++;
					break;

				case 2:
					indexes[tempi] = tetext[i * 9 + 1];
					indexesedge[tempiedge] = tetext[i * 9 + 1];
					tempiedge++;
					indexesedge[tempiedge] = tetext[i * 9 + 4];
					tempiedge++;
					tempi++;
					indexes[tempi] = tetext[i * 9 + 4];
					indexesedge[tempiedge] = tetext[i * 9 + 4];
					tempiedge++;
					indexesedge[tempiedge] = tetext[i * 9 + 2];
					tempiedge++;
					tempi++;
					indexes[tempi] = tetext[i * 9 + 2];
					indexesedge[tempiedge] = tetext[i * 9 + 2];
					tempiedge++;
					indexesedge[tempiedge] = tetext[i * 9 + 1];
					tempiedge++;
					tempi++;
					break;

				case 3:
					indexes[tempi] = tetext[i * 9 + 2];
					indexesedge[tempiedge] = tetext[i * 9 + 2];
					tempiedge++;
					indexesedge[tempiedge] = tetext[i * 9 + 3];
					tempiedge++;
					tempi++;
					indexes[tempi] = tetext[i * 9 + 3];
					indexesedge[tempiedge] = tetext[i * 9 + 3];
					tempiedge++;
					indexesedge[tempiedge] = tetext[i * 9 + 4];
					tempiedge++;
					tempi++;
					indexes[tempi] = tetext[i * 9 + 4];
					indexesedge[tempiedge] = tetext[i * 9 + 4];
					tempiedge++;
					indexesedge[tempiedge] = tetext[i * 9 + 2];
					tempiedge++;
					tempi++;
					break;

				}
			}
		}
	}
	i = 0; j = 0; k = 0;
	/*Write the index of exterior triangles info to a file*/
	std::ofstream trianglesfile("ext_Triangles.txt");
	for (i = 0; i < ntriangles; i++)
	{
		trianglesfile << "ID: " << i << "   " << indexes[i * 3] << " , " << indexes[i * 3 + 1] << " , " << indexes[i * 3 + 2] << std::endl;
	}
	trianglesfile.close();
	std::cout << "Index array for model drawing construction complete" << std::endl;

	/*read the interior tet info*/
	std::cout << "Reading interior tet info from file" << std::endl;
	if (!nc_inq_dimid(ncid, "tetinterior", &tempid2))//get the id of the dimension which name is tetinterior.(which represent the number of interior tetrahedron)
	{
		if ((retval = nc_inq_dimlen(ncid, tempid2, &templ2)))//get the length of the dimension that represent the interior tet.
			ERR(retval);
		nint = templ2;
		tetint = new int[nint * 5];//no need to specify the surface info here, so only 5 entries is needed, first one for material, following four for node id. 
		if ((retval = nc_inq_varid(ncid, "tetrahedron_interior", &tempid2)))//get the id of the variable that represent the exterior tetrahedron.
			ERR(retval);
		if ((retval = nc_get_var_int(ncid, tempid2, tetint)))
			ERR(retval);
		std::cout << "Interior tet info readin complete;" << std::endl;
	}
	else
	{
		std::cout << "None interior tet found" << std::endl;
	}

	/*Generate the tet node structure, 14 entries:
	first entry is the volume group id,
	following four are node indexes, 
	next four are surface type(-1 means shared with neighbor, and other numbers represent the sideset id),
	next one is the id where the tet is located in a easier to search straight mesh, the ID is calculated as sizeofx*sizeofy*Z+sizeofx*Y+X
	next four are neighbors
	*/
	nall = next + nint;
	//N_par = nall/4;
	tetall = new int[nall * tetsize];
	for (i = 0; i < nall*tetsize; i++)
	{
		tetall[i] = -1;
	}
	double3 tempnodes[4]; //the temp variable to store the nodes of the tet
	double *tempvolume;
	volume = (double*)malloc(sizeof(double)*nall);

	/*tempvolume = (double*)malloc(sizeof(double));
	*tempvolume = 0.0;
	tetvolumeHost({ -5.0, 5.0, -5.0}, { 5.0, 5.0, -5.0}, { 5.0, 5.0, 5.0 }, { 0.05, 0.3474, -0.226}, tempvolume);
	std::cout << *tempvolume << std::endl;*/
	
	for(i =0;i<next;i++)//move the data from tetext to tetall, since tetext has a period of 9 and tetall has a period of 13, we have to do some trick to put everything in line;
	{
		for (j = 0; j < 9; j++)
		{
			tetall[i*tetsize + j] = tetext[i * 9 + j];
			if (j>0 && j < 5)
			{
				tempnodes[j - 1].x = nodes[tetall[i*tetsize + j]*3];
				tempnodes[j - 1].y = nodes[tetall[i*tetsize + j]*3 + 1];
				tempnodes[j - 1].z = nodes[tetall[i*tetsize + j]*3 + 2];
			}
		}
		tetvolumeHost(tempnodes[0], tempnodes[1], tempnodes[2], tempnodes[3], (volume+i));
    }
	
	for (i = 0; i<nint; i++)//similarly for tetint
	{
		for (j = 0; j < 5; j++)
		{
			tetall[(i + next) * tetsize + j] = tetint[i * 5 + j];
			if (j>0 && j < 5)
			{
				tempnodes[j - 1].x = nodes[tetall[(i + next)*tetsize + j]*3];
				tempnodes[j - 1].y = nodes[tetall[(i + next)*tetsize + j]*3 + 1];
				tempnodes[j - 1].z = nodes[tetall[(i + next)*tetsize + j]*3 + 2];
			}
		}
		tetvolumeHost(tempnodes[0], tempnodes[1], tempnodes[2], tempnodes[3], volume+i+next);
	}
	

	/*Link the tet mesh together*/
	//ConnectTetHost(tetall, tetsize, nall);
	ConnectTetHost_2(tetall, tetsize, nall,nnodes);
	/*std::cout << "Tet mesh Linked!" << std::endl;
	std::ofstream meshfile("mesh.txt");
	for (i = 0; i < nall; i++)
	{
		meshfile << i << ":  ";
		for (j = 0; j < tetsize; j++)
			meshfile << tetall[i*tetsize + j] << ";";
		meshfile << std::endl;
	}*/
	/*send the volume info to device*/
	cudaMalloc(&dvolume, sizeof(double)*nall);
	showcudaerror();
	cudaMemcpy(dvolume, volume, sizeof(double)*nall, cudaMemcpyHostToDevice);
	showcudaerror();
	/*send the tet mesh info to device*/
	cudaMalloc(&dtetall, sizeof(int)*nall*tetsize);
	showcudaerror();
	cudaMemcpy(dtetall, tetall, sizeof(int)*nall*tetsize, cudaMemcpyHostToDevice);
	showcudaerror();
		
	/*Allocate the memory to store the info of where the partiles are located in tet mesh*/
	meshindextet_temp = (int *)malloc(sizeof(int)*nall);
	for (i = 0; i < nall; i++)
	{
		meshindextet_temp[i] = -1;
	}
	activetet(nodes, xrange, yrange, zrange, norm, tetall, meshindextet_temp, &N_par, nall, tetsize);

	meshindextet = (int *)malloc(sizeof(int)*N_par);
	meshindextet_shuffle = (int *)malloc(sizeof(int)*N_par);
	for (i = 0; i < N_par; i++)
	{
		meshindextet[i] = -1;
	}
	for (i = 0; i < N_par / phase_step; i++)//change the tet index every phase_step numbers in meshindextet 
	{
		for (j = 0; j < phase_step; j++)
		{
			meshindextet[i*phase_step + j] = meshindextet_temp[i];
		}
	}

	cudaMalloc(&dmeshindextet, sizeof(int)*N_par);
	showcudaerror();
	cudaMemcpy(dmeshindextet, meshindextet, sizeof(int)*N_par, cudaMemcpyHostToDevice);
	showcudaerror();


	/*Read the field info from file*/
	std::cout << "Reading the field info from file2;" << std::endl;
	if ((retval = nc_open(FILE_NAME2, NC_NOWRITE, &ncid)))
	{
		ERR(retval);
		std::cout << std::endl;
		std::cout << "Press Enter to Exit";
		std::getline(std::cin, exitchar);
		exit(ERRCODE);
	}
	printf("The File ID is: %d\n", ncid);
	
	/*read the eField info*/
	Efd = new double[nnodes * 3];//construct the array for the Efield.
	Efd_img = new double[nnodes * 3];//construct the array for the imaginary part of Efield.
	for (i = 0; i < nnodes * 3; i++)
	{
		Efd[i] = 0;
		Efd_img[i] = 0;
	}
	if ((retval = nc_inq_varid(ncid, "efield", &tempid1)))//get the id of the variable that represent the efield.
	{
		ERR(retval);
	}
	else if ((retval = nc_get_var_double(ncid, tempid1, Efd)))//read in the value of efield on each node
	{
		ERR(retval);
	}
	if ((retval = nc_inq_varid(ncid, "efield_imag", &tempid1)))//get the id of the variable that represent the imaginary part of the efield.
	{
		ERR(retval);
	}
	else if ((retval = nc_get_var_double(ncid, tempid1, Efd_img)))//read in the value of efield on each node
	{
		ERR(retval);
	}

	/*read the bField info*/
	Bfd = new double[nnodes * 3];//construct the array for the Efield.
	Bfd_img = new double[nnodes * 3];//construct the array for the imaginary part of Efield.
	for (i = 0; i < nnodes * 3; i++)
	{
		Bfd[i] = 0;
		Bfd_img[i] = 0;
	}
	if ((retval = nc_inq_varid(ncid, "bfield", &tempid1)))//get the id of the variable that represent the bfield.
	{
		ERR(retval);
	}
	else if ((retval = nc_get_var_double(ncid, tempid1, Bfd)))//read in the value of bfield on each node
	{
		ERR(retval);
	}
	if ((retval = nc_inq_varid(ncid, "bfield_imag", &tempid1)))//get the id of the variable that represent the imaginary part of bfield.
	{
		ERR(retval);
	}
	else if ((retval = nc_get_var_double(ncid, tempid1, Bfd_img)))//read in the value of imag bfield
	{
		ERR(retval);
	}
	if ((retval = nc_inq_varid(ncid, "frequency", &tempid1)))//get the id of the variable that represent the frequency.
	{
		if ((retval = nc_inq_varid(ncid, "frequencyreal", &tempid1)))
		{
			ERR(retval);
		}
		else if ((retval = nc_get_var_double(ncid, tempid1, &fre)))//read in the value of imag bfield
		{
			ERR(retval);
		}
	}
	else if ((retval = nc_get_var_double(ncid, tempid1, &fre)))//read in the value of imag bfield
	{
		ERR(retval);
	}
	
	///*print field*/
	/*std::ofstream efieldfile("Efield.txt");
	for (i = 0; i < nnodes * 3; i++)
	{
		efieldfile<< Efd[i] << "	;	" << Efd_img[i] << std::endl;
	}
	efieldfile.close();
	std::ofstream bfieldfile("Bfield.txt");

	for (i = 0; i < nnodes * 3; i++)
	{
		bfieldfile << Bfd[i] << "	;	" << Bfd_img[i] << std::endl;
	}
	bfieldfile.close();*/
	
	/*put the field together so we don't need to copy four arraies of field to GPU, most of the case only one part is non_zero so we can do the following instead of finding the average*/
	Efd_temp = (double3 *)malloc(sizeof(double3)*nnodes);
	Bfd_temp = (double3 *)malloc(sizeof(double3)*nnodes);
	if (Bfd_img[0] == 0)
	{
		for (i = 0; i < nnodes; i++)
		{
			/*std::cout << Bfd[i * 3] <<" "<<Efd[i*3]<< std::endl;
			std::cout << Bfd_img[i * 3] << " " << Efd_img[i * 3] << std::endl;
			std::cout << Bfd[i * 3 + 1] << " " << Efd[i * 3+1] << std::endl;
			std::cout << Bfd_img[i * 3 + 1] << " " << Efd_img[i * 3 + 1] << std::endl;
			std::cout << Bfd[i * 3 + 2] << " " << Efd[i * 3 + 2] << std::endl;
			std::cout << Bfd_img[i * 3 + 2] << " " << Efd_img[i * 3 + 2] << std::endl;
			std::cout << std::endl;
			Efd_temp[i].x = sqrt(Efd[i * 3] * Efd[i * 3] + Efd_img[i * 3] * Efd_img[i * 3])*((Efd[i*3]>0)-(Efd[i*3]<0));
			Efd_temp[i].y = sqrt(Efd[i * 3 + 1] * Efd[i * 3 + 1] + Efd_img[i * 3 + 1] * Efd_img[i * 3 + 1]) * ((Efd[i * 3 + 1]>0) - (Efd[i * 3 + 1]<0));
			Efd_temp[i].z = sqrt(Efd[i * 3 + 2] * Efd[i * 3 + 2] + Efd_img[i * 3 + 2] * Efd_img[i * 3 + 2]) * ((Efd[i * 3 + 2]>0) - (Efd[i * 3 + 2]<0));
			Bfd_temp[i].x = sqrt(Bfd[i * 3] * Bfd[i * 3] + Bfd_img[i * 3] * Bfd_img[i * 3])*((Bfd[i * 3]>0) - (Bfd[i * 3]<0));
			Bfd_temp[i].y = sqrt(Bfd[i * 3 + 1] * Bfd[i * 3 + 1] + Bfd_img[i * 3 + 1] * Bfd_img[i * 3 + 1])*((Bfd[i * 3+1]>0) - (Bfd[i * 3+1]<0));
			Bfd_temp[i].z = sqrt(Bfd[i * 3 + 2] * Bfd[i * 3 + 2] + Bfd_img[i * 3 + 2] * Bfd_img[i * 3 + 2])*((Bfd[i * 3+2]>0) - (Bfd[i * 3+2]<0));
			*/
			Efd_temp[i].x = Efd[i * 3];
			Efd_temp[i].y = Efd[i * 3 + 1];
			Efd_temp[i].z = Efd[i * 3 + 2];
			Bfd_temp[i].x = Bfd[i * 3];
			Bfd_temp[i].y = Bfd[i * 3 + 1];
			Bfd_temp[i].z = Bfd[i * 3 + 2];
			//std::cout << Bfd_temp[i].x << std::endl;
		}
	}
	else
	{
		for (i = 0; i < nnodes; i++)
		{
			/*std::cout << Bfd[i * 3] <<" "<<Efd[i*3]<< std::endl;
			std::cout << Bfd_img[i * 3] << " " << Efd_img[i * 3] << std::endl;
			std::cout << Bfd[i * 3 + 1] << " " << Efd[i * 3+1] << std::endl;
			std::cout << Bfd_img[i * 3 + 1] << " " << Efd_img[i * 3 + 1] << std::endl;
			std::cout << Bfd[i * 3 + 2] << " " << Efd[i * 3 + 2] << std::endl;
			std::cout << Bfd_img[i * 3 + 2] << " " << Efd_img[i * 3 + 2] << std::endl;
			std::cout << std::endl;*/
			Efd_temp[i].x = sqrt(Efd[i * 3] * Efd[i * 3] + Efd_img[i * 3] * Efd_img[i * 3])*((Efd[i*3]>0)-(Efd[i*3]<0));
			Efd_temp[i].y = sqrt(Efd[i * 3 + 1] * Efd[i * 3 + 1] + Efd_img[i * 3 + 1] * Efd_img[i * 3 + 1]) * ((Efd[i * 3 + 1]>0) - (Efd[i * 3 + 1]<0));
			Efd_temp[i].z = sqrt(Efd[i * 3 + 2] * Efd[i * 3 + 2] + Efd_img[i * 3 + 2] * Efd_img[i * 3 + 2]) * ((Efd[i * 3 + 2]>0) - (Efd[i * 3 + 2]<0));
			Bfd_temp[i].x = sqrt(Bfd[i * 3] * Bfd[i * 3] + Bfd_img[i * 3] * Bfd_img[i * 3])*((Bfd_img[i * 3]<0) - (Bfd_img[i * 3]>0));
			Bfd_temp[i].y = sqrt(Bfd[i * 3 + 1] * Bfd[i * 3 + 1] + Bfd_img[i * 3 + 1] * Bfd_img[i * 3 + 1])*((Bfd_img[i * 3 + 1]<0) - (Bfd_img[i * 3 + 1]>0));
			Bfd_temp[i].z = sqrt(Bfd[i * 3 + 2] * Bfd[i * 3 + 2] + Bfd_img[i * 3 + 2] * Bfd_img[i * 3 + 2])*((Bfd_img[i * 3 + 2]<0) - (Bfd_img[i * 3 + 2]>0));
			
		
			//std::cout << Bfd_temp[i].x << std::endl;
		}
	}
	
	//normalize the fields to the maximum efield
	double efield_max = 0;
	for (i = 0; i < nnodes; i++)
	{
		efield_max = efield_max>(Efd_temp[i].x*Efd_temp[i].x + Efd_temp[i].y*Efd_temp[i].y + Efd_temp[i].z*Efd_temp[i].z) ? efield_max : (Efd_temp[i].x*Efd_temp[i].x + Efd_temp[i].y*Efd_temp[i].y + Efd_temp[i].z*Efd_temp[i].z);
	}
	efield_max = sqrt(efield_max);

	std::cout << efield_max << std::endl;
	for (i = 0; i < nnodes; i++)
	{
		Efd_temp[i].x = Efd_temp[i].x / efield_max;
		Efd_temp[i].y = Efd_temp[i].y / efield_max;
		Efd_temp[i].z = Efd_temp[i].z / efield_max;
		Bfd_temp[i].x = Bfd_temp[i].x / efield_max;
		Bfd_temp[i].y = Bfd_temp[i].y / efield_max;
		Bfd_temp[i].z = Bfd_temp[i].y / efield_max;
	}
	///*print the normalized field*/
	/*efieldfile.open("Efield_norm.txt", std::ios::out);
	for (i = 0; i < nnodes; i++)
	{
		efieldfile << Efd_temp[i].x << "	;	" << Efd_temp[i].y << "   ;   " << Efd_temp[i].z<< std::endl;
	}
	efieldfile.close();
	bfieldfile.open("Bfield_norm.txt", std::ios::out);

	for (i = 0; i < nnodes; i++)
	{
		bfieldfile << Bfd_temp[i].x << "	;	" << Bfd_temp[i].y << "   ;   " << Bfd_temp[i].z << std::endl;
	}
	bfieldfile.close();*/

	cudaMalloc(&dEfd, sizeof(double3) * nnodes);
	showcudaerror();
	cudaMalloc(&dBfd, sizeof(double3) * nnodes);
	showcudaerror();

	cudaMemcpy(dEfd, Efd_temp, sizeof(double3)*nnodes,cudaMemcpyHostToDevice);
	showcudaerror();
	cudaMemcpy(dBfd, Bfd_temp, sizeof(double3)*nnodes,cudaMemcpyHostToDevice);
	showcudaerror();
	std::cout << "Field info readin complete;" << std::endl;

	initphase = (int*)malloc(sizeof(int)*N_par);
	initphase_shuffle = (int*)malloc(sizeof(int)*N_par);
	cudaMalloc(&dinitphase, sizeof(int)*N_par);
	showcudaerror();

	pArgc = &argc;
	pArgv = argv;
	
#if defined(__linux__)
	setenv("DISPLAY", ":0", 0);
#endif

	printf("%s starting...\n", sSDKsample);

	if (argc > 1)
	{
		if (checkCmdLineFlag(argc, (const char **)argv, "file"))
		{
			// In this mode, we are running non-OpenGL and doing a compare of the VBO was generated correctly
			getCmdLineArgumentString(argc, (const char **)argv, "file", (char **)&ref_file);
		}
	}

	printf("\n");
	/*Now that we have node, tet and field info, we will allocate the impact related array*/
	impact = new int[N_par];
	impact_shuffle = new int[N_par];
	cudaMalloc(&dimpact, N_par*sizeof(int));
	showcudaerror();
	impactenergy = new double[N_par*N_cycles * 2];
	impactenergy_shuffle = new double[N_par*N_cycles * 2];
	cudaMalloc(&D_impactenergy, N_par*N_cycles * 2 * sizeof(double));
	showcudaerror();

	flag = new int[N_par];
	flag_shuffle = new int[N_par];
	cudaMalloc(&dflag, N_par*sizeof(int));
	showcudaerror();
	
	Hposition = (double3*)malloc(N_par*sizeof(double3));
	Hvelocity = (double3*)malloc(N_par*sizeof(double3));
	barycentric = (double4*)malloc(sizeof(double4)*N_par);
	H_position0 = (double3*)malloc(sizeof(double3)*N_par);
	H_momentumf = (double3*)malloc(sizeof(double3)*N_par);
	Hposition_shuffle = (double3*)malloc(N_par*sizeof(double3));
	Hvelocity_shuffle = (double3*)malloc(N_par*sizeof(double3));
	barycentric_shuffle = (double4*)malloc(sizeof(double4)*N_par);
	H_position0_shuffle = (double3*)malloc(sizeof(double3)*N_par);
	H_momentumf_shuffle = (double3*)malloc(sizeof(double3)*N_par);

	cudaMalloc(&dbarycentric, N_par*sizeof(double4));
	cudaMalloc(&d_position0, N_par*sizeof(double3));
	cudaMalloc(&d_momentumf, N_par*sizeof(double3));
	cudaMalloc(&D_momentumt, N_par*sizeof(double3));
	
	cudaMalloc(&D_p_Efd, N_par*sizeof(double3));
	cudaMalloc(&D_p_Bfd, N_par*sizeof(double3));
	//cudaMalloc(&D_p_nodes, N_par*sizeof(int4));
	
	showcudaerror();

	i = 0;
	j = 0;
	k = 0;
	runTest(argc, argv, ref_file);

	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();
	printf("%s completed, returned %s\n", sSDKsample, (g_TotalErrors == 0) ? "OK" : "ERROR!");
	exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}

void computeFPS()
{
	frameCount++;
	fpsCount++;

	if (fpsCount == fpsLimit)
	{
		avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		fpsCount = 0;
		fpsLimit = (int)MAX(avgFPS, 1.f);

		sdkResetTimer(&timer);
	}

	char fps[256];
	sprintf(fps, "Cuda GL Interop (VBO): %3.1f fps (Max 60Hz); Master Clock Time: %3.10f (unit: # of Cycles);N_par:%d", avgFPS, g_fAnim,N_par);
	glutSetWindowTitle(fps);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("Cuda GL Interop (VBO)");
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMotionFunc(motion);
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

	// initialize necessary OpenGL extensions
	glewInit();

	if (!glewIsSupported("GL_VERSION_2_0 "))
	{
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}

	// default initialization
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// viewport
	glViewport(0, 0, window_width, window_height);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height, 0.01, 100.0);

	SDK_CHECK_ERROR_GL();

	return true;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
bool runTest(int argc, char **argv, char *ref_file)
{
	// Create the CUTIL timer
	sdkCreateTimer(&timer);

	// command line mode only
	if (ref_file != NULL)
	{
		// This will pick the best possible CUDA capable device
		int devID = findCudaDevice(argc, (const char **)argv);
		// create VBO
		checkCudaErrors(cudaMalloc((void **)&d_vbo_buffer, N_par * 3 * sizeof(double)));
		checkCudaErrors(cudaMalloc((void **)&d_vbov_buffer, N_par * 3 * sizeof(double)));
		// run the cuda part
		runAutoTest(devID, argv, ref_file);
		// check result of Cuda step
		checkResultCuda(argc, argv, vbo);
		cudaFree(d_vbo_buffer);
		cudaFree(d_vbov_buffer);
		d_vbo_buffer = NULL;
		d_vbov_buffer = NULL;
	}
	else
	{
		// First initialize OpenGL context, so we can properly set the GL for CUDA.
		// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
		if (false == initGL(&argc, argv))
		{
			return false;
		}
		// use command-line specified CUDA device, otherwise use device with highest Gflops/s
		if (checkCmdLineFlag(argc, (const char **)argv, "device"))
		{
			if (gpuGLDeviceInit(argc, (const char **)argv) == -1)
			{
				return false;
			}
		}
		else
		{
			cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
		}
		// register callbacks
		glutDisplayFunc(display);
		glutKeyboardFunc(keyboard);
		glutMouseFunc(mouse);
		glutMotionFunc(motion);
		// create VBO
		createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsNone);
		createVBO(&vbov, &cuda_vbov_resource, cudaGraphicsMapFlagsNone);
		// run the cuda part
		runCuda(&cuda_vbo_resource, &cuda_vbov_resource, 0);
		std::cout << "Initalize particles finish" << std::endl;
/*
		runCuda(&cuda_vbo_resource, &cuda_vbov_resource, 1);
		cudaDeviceReset();
		exit(0);
		*/

		glutMainLoop();
	}
	return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource **vbo_resource, struct cudaGraphicsResource **vbov_resource, int func)
{
	// map OpenGL buffer object for writing from CUDA
	double3 *pptr;
	double3 *vptr;
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
	checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
	size_t num_bytes;

	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&pptr, &num_bytes,*vbo_resource));
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
	checkCudaErrors(cudaGraphicsMapResources(1, vbov_resource, 0));
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
	size_t num_bytesv;
	
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&vptr, &num_bytes,*vbov_resource));
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
	

	launch_kernel(pptr, vptr, g_fAnim, func);

	//cudaDeviceSynchronize();

	// unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, vbov_resource, 0));
}


void sdkDumpBin2(void *data, unsigned int bytes, const char *filename)
{
	printf("sdkDumpBin: <%s>\n", filename);
	FILE *fp;
	FOPEN(fp, filename, "wb");
	fwrite(data, bytes, 1, fp);
	fflush(fp);
	fclose(fp);
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runAutoTest(int devID, char **argv, char *ref_file)
{
	char *reference_file = NULL;
	void *imageData = malloc(N_par*sizeof(double3));

	// execute the kernel
	launch_kernel((double3 *)d_vbo_buffer, (double3 *)d_vbov_buffer, g_fAnim, 0);

	//cudaDeviceSynchronize();
	getLastCudaError("launch_kernel failed");

	checkCudaErrors(cudaMemcpy(imageData, d_vbo_buffer, N_par*sizeof(double3), cudaMemcpyDeviceToHost));

	//sdkDumpBin2(imageData, mesh_width*mesh_height*sizeof(float), "simpleGL.bin");
	reference_file = sdkFindFilePath(ref_file, argv[0]);

	if (reference_file &&
		!sdkCompareBin2BinFloat("simpleGL.bin", reference_file,
		N_par*sizeof(double3),
		MAX_EPSILON_ERROR, THRESHOLD, pArgv[0]))
	{
		g_TotalErrors++;
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
	unsigned int vbo_res_flags)
{
	assert(vbo);

	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	unsigned int size = N_par * 3 * sizeof(double);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

	//glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

	// create buffer object
	glGenBuffers(1, &nvbo);
	glBindBuffer(GL_ARRAY_BUFFER, nvbo);
	// initialize buffer object
	glBufferData(GL_ARRAY_BUFFER, sizeof(double)*nnodes * 3, nodesdisp, GL_DYNAMIC_DRAW);

	// create buffer object
	glGenBuffers(1, &vboindex);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboindex);
	// initialize buffer object
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * ntriangles * 3, indexes, GL_DYNAMIC_DRAW);

	// create buffer object
	glGenBuffers(1, &vboindexedg);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboindexedg);
	// initialize buffer object
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * ntriangles * 6, indexesedge, GL_DYNAMIC_DRAW);

	SDK_CHECK_ERROR_GL();
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{

	//// unregister this buffer object with CUDA
	//cudaGraphicsUnregisterResource(vbo_res);

	//glBindBuffer(1, *vbo);
	//glDeleteBuffers(1, vbo);
	//glBindBuffer(1, nvbo);
	//glDeleteBuffers(1, &nvbo);
	//*vbo = 0;
	//nvbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
	sdkStartTimer(&timer);

	// run CUDA kernel to generate vertex positions
	runCuda(&cuda_vbo_resource, &cuda_vbov_resource, 1);
	//cudaDeviceSynchronize();
	//cudaStreamSynchronize(0);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(translate_x, translate_y, translate_z);
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y, 0.0, 1.0, 0.0);

	// render from the vbo
	glPointSize(4.0);

	glEnableClientState(GL_VERTEX_ARRAY);

	glColor3f(0.0, 0.0, 1.0);

	// create buffer object

	glBindBuffer(GL_ARRAY_BUFFER, nvbo);
	glVertexPointer(3, GL_DOUBLE, 0, 0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboindex);
	//draw the surfaces
	glDrawElements(GL_TRIANGLES, ntriangles * 3, GL_UNSIGNED_INT, 0);
	////draw the vertexes
	//glPointSize(5.0);
	//glColor3f(0.0, 1.0, 0.0);
	//glDrawArrays(GL_POINTS, 0, nnodes);

	//draw the edges
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboindexedg);
	glColor3f(1.0, 0.0, 1.0);
	glDrawElements(GL_LINES, ntriangles * 6, GL_UNSIGNED_INT, 0);


	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(3, GL_DOUBLE, 0, 0);
	glColor3f(1.0, 0.0, 0.0);
	glDrawArrays(GL_POINTS, 0, N_par);
	glDisableClientState(GL_VERTEX_ARRAY);

	glutSwapBuffers();

	g_fAnim += dt;

	sdkStopTimer(&timer);
	computeFPS();
	
	if (g_fAnim >= N_cycles)
	{
		int i, j;
		
		g_fAnim = 0;
		
		if (fdnorm > fdnorm_max)
		{
			exit(3);
		}
		cudaMemcpy(flag, dflag, sizeof(int)*N_par, cudaMemcpyDeviceToHost);
		cudaMemcpy(impactenergy, D_impactenergy, sizeof(double)*N_par*N_cycles*2, cudaMemcpyDeviceToHost);
		cudaMemcpy(impact, dimpact, sizeof(int)*N_par, cudaMemcpyDeviceToHost);
		cudaMemcpy(Hposition, d_vbo_buffer, sizeof(double3)*N_par, cudaMemcpyDeviceToHost);
		std::ofstream flagfile("c:/resultsfortracking/scan/"+ std::to_string((int)fdnorm) + "flag" + ".bin", std::ios::binary);
		std::ofstream energyfile("c:/resultsfortracking/scan/" +  std::to_string((int)fdnorm) + "momentum" + ".bin", std::ios::binary);
		std::ofstream impactfile("c:/resultsfortracking/scan/" +  std::to_string((int)fdnorm) + "impact" + ".bin", std::ios::binary);
		std::ofstream positionfile("c:/resultsfortracking/scan/" + std::to_string((int)fdnorm) + "lastposition" + ".bin", std::ios::binary);
		/*flagfile.write((const char*)flag, N_par*sizeof(int));
		impactfile.write((const char*)impact, N_par*sizeof(int));
		energyfile.write((const char*)impactenergy, N_par*sizeof(double)*N_cycles*2);*/
		for (i = 0; i < N_par; i++)
		{
			positionfile << Hposition[i].x*norm << "  " << Hposition[i].y*norm << "  " << Hposition[i].z*norm << std::endl;

			flagfile << "ID="<<i<<": "<<flag[i] << std::endl;
			impactfile << "ID=" << i << ": " << impact[i] << std::endl;	
			energyfile << "ID = " << i << ":" << std::endl;
			for (j = 0; j < N_cycles * 2; j++)
			{
				energyfile << impactenergy[i*N_cycles * 2 + j] << std::endl;
			}
		}
		flagfile.close();
		impactfile.close();
		energyfile.close();
		fdnorm += fdnorm_step;// *(initphase / 100);
		activetet(nodes, xrange, yrange, zrange, norm, tetall, meshindextet_temp, &N_par, nall, tetsize);
		runCuda(&cuda_vbo_resource, &cuda_vbov_resource, 0);//reset the particles location, momentums, flags and impactmomentums.
	}
}

void timerEvent(int value)
{
	glutPostRedisplay();
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

void cleanup()
{
	sdkDeleteTimer(&timer);

	if (vbo)
	{
		deleteVBO(&vbo, cuda_vbo_resource);
	}
	if (nvbo)
	{
		deleteVBO(&nvbo, cuda_vbo_resource);
	}
}


////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key)
	{
	case (27) :
		exit(EXIT_SUCCESS);
		break;
	case('r') :						//R - reset view
		/*rotate_x = 0;
		rotate_y = 0;
		translate_z = -3.0;*/
		runCuda(&cuda_vbo_resource, &cuda_vbov_resource, 0);
		break;
	case('a') :						//a - increase field level
		fdnorm += 1.0e4;
		
		printf("Max Efield = %2.3e V/m \n", fdnorm);
		break;
	case('s') :						//s - decrease field level
		fdnorm -= 1.0e4;
		
		printf("Max Efield = %2.3e V/m \n", fdnorm);
		break;
	case('d') :						//d - shut down field level
		fdnorm = 0.0;
		
		printf("Max Efield = %2.3e V/m \n", fdnorm);
		break;
	case('q') :						//q - decrease dt
		dt += 1.0e-3;
		printf("dt = %2.3f * T\n", dt);
		break;
	case('w') :						//w - decrease dt
		dt -= 1.0e-3;
		printf("dt = %2.3f * T\n", dt);
		break;
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_buttons |= 1 << button;
	}
	else if (state == GLUT_UP)
	{
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - mouse_old_x);
	dy = (float)(y - mouse_old_y);

	if (mouse_buttons & 1)
	{
		rotate_x += dy * 0.1f;
		rotate_y += dx * 0.1f;
	}
	else if (mouse_buttons & 4)
	{
		translate_z += dy * 0.001f;
	}
	else if (mouse_buttons & 2)
	{
		translate_x += dx*0.0005f;
		translate_y -= dy*0.0005f;
	}
	mouse_old_x = x;
	mouse_old_y = y;
}

////////////////////////////////////////////////////////////////////////////////
//! Check if the result is correct or write data to file for external
//! regression testing
////////////////////////////////////////////////////////////////////////////////
void checkResultCuda(int argc, char **argv, const GLuint &vbo)
{
	if (!d_vbo_buffer)
	{
		cudaGraphicsUnregisterResource(cuda_vbo_resource);

		// map buffer object
		glBindBuffer(GL_ARRAY_BUFFER_ARB, vbo);
		float *data = (float *)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);

		// check result
		if (checkCmdLineFlag(argc, (const char **)argv, "regression"))
		{
			// write file for regression test
			/*sdkWriteFile<float>("./data/regression.dat",
				data, mesh_width * mesh_height * 3, 0.0, false);*/
		}

		// unmap GL buffer object
		if (!glUnmapBuffer(GL_ARRAY_BUFFER))
		{
			fprintf(stderr, "Unmap buffer failed.\n");
			fflush(stderr);
		}

		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo,
			cudaGraphicsMapFlagsWriteDiscard));

		SDK_CHECK_ERROR_GL();
	}
}

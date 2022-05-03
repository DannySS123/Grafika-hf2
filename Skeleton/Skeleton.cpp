//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Szaraz Daniel
// Neptun : GT5X34
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

const int maxdepth = 10;
const float epsilon = 0.005f;

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd* M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
	virtual void Animate(float dt) = 0;
};

vec4 quat(vec3 axis, float angle) {
	vec3 res = normalize(axis) * sinf(angle / 2);
	return vec4(res.x, res.y, res.z, cosf(angle/2));
}

vec4 quatMul(vec4 q1, vec4 q2) {
	vec3 q1xyz = vec3(q1.x, q1.y, q1.z);
	vec3 q2xyz = vec3(q2.x, q2.y, q2.z);
	vec3 res = q1.w * q2xyz + q2.w * q1xyz + cross(q1xyz, q2xyz);
	return vec4(
		res.x, res.y, res.z,
		q1.w * q2.w - dot(q1xyz, q2xyz)
	);
}

vec4 quatInv(vec4 q) {
	return vec4(-q.x, -q.y, -q.z, q.w);
}

vec3 quatRot(vec4 q, vec3 p) {
	vec4 qInv = quatInv(q);
	vec4 res = quatMul(quatMul(q, vec4(p.x, p.y, p.z, 0)), qInv);
	return vec3(res.x, res.y, res.z);
}

struct Sphere : public Intersectable {
	vec3 center;
	float radius;

	Sphere(const vec3& _center, float _radius, Material* _material) {
		center = _center;
		radius = _radius;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0f;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) * (1.0f / radius);
		hit.material = material;
		return hit;
	}

	void Animate(float dt) {
		center.y += dt/50;
	}
};

struct Plane : public Intersectable {
	vec3 normal, point;
	Plane(const vec3& n, const vec3& p, Material* _material) {
		normal = n;
		point = p;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		float t = -(dot(normal, ray.start) - dot(normal, point)) / (dot(normal, ray.dir));
		hit.t = t;
		hit.position = ray.start + (ray.dir * hit.t);
		hit.normal = normal;
		hit.material = material;
		return hit;
	}

	void Animate(float dt) {

	}
};

struct Circle : public Intersectable {
	vec3 normal, point;
	float radius;
	Circle(const vec3& n, const vec3& p, const float _radius,  Material* _material) {
		normal = n;
		point = p;
		radius = _radius;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		float t = -(dot(normal, ray.start) - dot(normal, point)) / (dot(normal, ray.dir));
		vec3 pos = ray.start + (ray.dir * t);
		if (length(point - pos) > radius) {
			return hit;
		}
		hit.t = t;
		hit.position = pos;
		hit.normal = normal;
		hit.material = material;
		return hit;
	}

	void Animate(float dt) {

	}
};

struct Cylinder : public Intersectable {
	vec3 bottom, middle, top;
	float radius, heigth;

	Cylinder(const vec3& _bottom, const vec3& _top, const float _radius, Material* _material) {
		bottom = _bottom;
		top = _top;
		radius = _radius;
		heigth = length(top-bottom);
		middle = bottom + (top - bottom) / 2;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;

		vec4 q = quat(vec3(0, 1, 0), M_PI / 4);
		//vec3 rd = quatRot(q, ray.dir);
		//vec3 rs = quatRot(q, ray.start);

		vec3 v0 = normalize(top-bottom);
		vec3 dv0dv0 = ray.dir - v0 * dot(ray.dir, v0);
		vec3 spv0 = ray.start - bottom - v0 * (dot(ray.start, v0) - dot(bottom, v0));
		float a = dot(dv0dv0, dv0dv0);
		float b = 2 * dot(dv0dv0, spv0);
		float c = dot(spv0, spv0) - radius*radius;

		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		vec3 pos1 = ray.start + ray.dir * t1;
		vec3 pos2 = ray.start + ray.dir * t2;
		vec3 m1 = bottom + v0 * (dot(pos1-bottom, top-bottom) / length(top-bottom));
		vec3 m2 = bottom + v0 * (dot(pos2-bottom, top-bottom) / length(top-bottom));

		vec3 M;
		if (length(m2 - middle) <=  heigth/2) {
			hit.position = pos2;
			hit.t = t2;
			M = m2;
		}
		else if (length(m1 - middle) <= heigth/2) {
			hit.position = pos1;
			hit.t = t1;
			M = m1;
		}
		else {
			return hit;
		}
		
		hit.normal = normalize(hit.position - M);
		hit.material = material;
		return hit;
	}

	void Animate(float dt) {
		bottom.y += dt;
		top.y += dt;
		heigth = length(top - bottom);
	}
};

struct Paraboloid : public Intersectable {
	vec3 normal, planePoint, focusPoint;

	Paraboloid(const vec3& _normal, const vec3& _planePoint, Material* _material) {
		normal = normalize(_normal);
		planePoint = _planePoint;
		focusPoint = planePoint + normal/15;
		material = _material;
	}

	Hit intersect(const Ray& ray) {

		vec4 q = quat(vec3(1,0,0), M_PI/2);
		//ray.dir = quatRot(q, ray.dir);
		//ray.start = quatRot(q, ray.start);

		Hit hit;

		float a = dot(ray.dir, ray.dir) - (dot(normal, ray.dir) * dot(normal, ray.dir));
		float b = 2 * (dot(ray.start, ray.dir) - dot(ray.dir, focusPoint)) - 2 * (dot(normal, ray.start - planePoint)*dot(normal,ray.dir));
		float c = dot(ray.start, ray.start) + dot(focusPoint, focusPoint) - 2 * dot(ray.start, focusPoint) -(dot(normal, ray.start-planePoint)* dot(normal, ray.start - planePoint));


		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;

		vec3 pos1 = ray.start + ray.dir * t1;
		vec3 pos2 = ray.start + ray.dir * t2;

		float l = length(planePoint - focusPoint)*4;

		if (t2 <= 0) {
			if (length(pos1 - focusPoint) <= l) {
				hit.t = t1;
				hit.position = pos1;
			}
			else {
				return hit;
			}
		}
		else {
			if (length(pos2 - focusPoint) <= l) {
				hit.t = t2;
				hit.position = pos2;
			}
			else if (length(pos1 - focusPoint) <= l) {
				hit.t = t1;
				hit.position = pos1;
			}
			else {
				return hit;
			}
		}

		vec3 newPos = hit.position + vec3(epsilon, 0, 0);
		float gx = fabs(dot(normal, newPos - planePoint)) - length(newPos - focusPoint);
		newPos = hit.position + vec3(0, epsilon, 0);
		float gy = fabs(dot(normal, newPos - planePoint)) - length(newPos - focusPoint);
		newPos = hit.position + vec3(0, 0, epsilon);
		float gz = fabs(dot(normal, newPos - planePoint)) - length(newPos - focusPoint);

		hit.normal = normalize(vec3(gx, gy, gz));


		hit.material = material;
		return hit;
	}

	void Animate(float dt) {

	}
};

class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye;
		lookat = _lookat;
		fov = _fov;
		vec3 w = eye - lookat;
		float windowSize = length(w) * tanf(fov / 2);
		right = normalize(cross(vup, w)) * windowSize;
		up = normalize(cross(w, right)) * windowSize;
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}

	void Animate(float dt) {
		vec3 d = eye - lookat;
		eye = vec3(d.x * cos(dt) + d.z * sin(dt), d.y, -d.x * sin(dt) + d.z * cos(dt)) + lookat;
		set(eye, lookat, up, fov);
	}
};

struct PointLight {
	vec3 location;
	vec3 power;

	PointLight(vec3 _location, vec3 _power) {
		location = _location;
		power = _power;
	}
	double distanceOf(vec3 point) {
		return length(location - point);
	}
	vec3 directionOf(vec3 point) {
		return normalize(location - point);
	}
	vec3 radianceAt(vec3 point) {
		double distance2 = dot(location - point, location - point);
		if (distance2 < epsilon) distance2 = epsilon;
		return power / distance2;
	}
};

float rnd() { return (float)rand() / RAND_MAX; }

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<PointLight*> lights;
	Camera camera;
	vec3 La;
public:
	void build() {
		vec3 eye = vec3(0, 0, 2), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.1f, 0.1f, 0.1f);
		vec3 Le(2, 2, 2);
		lights.push_back(new PointLight(vec3(1, 1, 0.2), Le));
		lights.push_back(new PointLight(vec3(0, 0.2, 0) + vec3(0, -0.01, 0.05), Le));

		vec3 kd(0.3f, 0.2f, 0.1f), ks(2, 2, 2);
		Material* material = new Material(kd, ks, 50);
		Material* m2 = new Material(vec3(0.1f, 0.2f, 0.3f), vec3(2, 2, 2), 50);
		objects.push_back(new Plane(vec3(0, 1, 0), vec3(0, -0.5, 0), m2));
		objects.push_back(new Cylinder(vec3(0,-0.5,0), vec3(0, -0.45, 0), 0.2, material));

		objects.push_back(new Circle(vec3(0,1,0), vec3(0, -0.45, 0), 0.2, material));
		objects.push_back(new Sphere(vec3(0, -0.45, 0), 0.04f, material));
		objects.push_back(new Cylinder(vec3(0, -0.45, 0), vec3(0, -0.1, -0.1), 0.02, material));
		objects.push_back(new Sphere(vec3(0, -0.1, -0.1), 0.04f, material));
		objects.push_back(new Cylinder(vec3(0, -0.1, -0.1), vec3(0, 0.2, 0), 0.02, material));
		objects.push_back(new Sphere(vec3(0, 0.2, 0), 0.04f, material));

		objects.push_back(new Paraboloid(vec3(0,-0.1,0.3), vec3(0,0.2,0), material));
	}

	void render(std::vector<vec4>& image) {
		long timeStart = glutGet(GLUT_ELAPSED_TIME);

		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
		int rt = glutGet(GLUT_ELAPSED_TIME) - timeStart;
		printf("Rendering time: %d ms\t FPS: %f\n", rt, 1000.0/rt);
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray, vec3 lightPos) {	// for directional lights
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray);
			if (hit.t > 0 && length(hit.position-ray.start) < length(ray.start-lightPos)) {
				return true;
			}
		}
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;
		vec3 outRadiance = hit.material->ka * La;
		for (PointLight* light : lights) {
			vec3 lhdir = normalize(light->location - hit.position);
			Ray shadowRay(hit.position + hit.normal * epsilon, lhdir);
			float cosTheta = dot(hit.normal, lhdir);
			if (cosTheta > 0 && !shadowIntersect(shadowRay, light->location)) {	// shadow computation
				outRadiance = outRadiance + light->radianceAt(hit.position) * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + lhdir);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) outRadiance = outRadiance + light->radianceAt(hit.position) * hit.material->ks * powf(cosDelta, hit.material->shininess);
			}
		}
		/*for (PointLight* light : lights) {	// Direct light source computation
			vec3 outDir = light->directionOf(hit.position);
			Hit shadowHit = firstIntersect(Ray(hit.position + hit.normal * epsilon, outDir));
			if (shadowHit.t < epsilon || shadowHit.t > light->distanceOf(hit.position)) {	// if not in shadow
				double cosThetaL = dot(hit.normal, outDir);
				if (cosThetaL >= epsilon) {
					outRadiance = outRadiance + hit.material->kd / M_PI * cosThetaL * light->radianceAt(hit.position);
				}
			}
		}*/

		return outRadiance;
	}
	/*
	// Find the first intersection of the ray with objects
	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (auto object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)) bestHit = hit;
		}
		return bestHit;
	}*/

	// Trace a ray and return the radiance of the visible surface
	/*vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);	// Find visible surface
		vec3 outRad(0, 0, 0);
		if (hit.t < 0 || depth >= maxdepth) return outRad;	// If there is no intersection

		vec3 N = hit.normal;	// normal of the visible surface
		vec3 outDir;
		for (auto light : lights) {	// Direct light source computation
			outDir = light->directionOf(hit.position);
			Hit shadowHit = firstIntersect(Ray(hit.position + N * epsilon, outDir));
			if (shadowHit.t < epsilon || shadowHit.t > light->distanceOf(hit.position)) {	// if not in shadow
				double cosThetaL = dot(N, outDir);
				if (cosThetaL >= epsilon) {
					outRad = outRad + hit.material->kd / M_PI * cosThetaL * light->radianceAt(hit.position);
				}
			}
		}

		double diffuseSelectProb = (hit.material->kd.x + hit.material->kd.y + hit.material->kd.z)/3;
		double mirrorSelectProb = (hit.material->ks.x + hit.material->ks.y + hit.material->ks.z)/3;

		//double rnd = random();	// Russian roulette to find diffuse, mirror or no reflection
		if (rnd() < diffuseSelectProb) { // diffuse
			double pdf = SampleDiffuse(N, ray.dir, outDir);
			double cosThetaL = dot(N, outDir);
			if (cosThetaL >= epsilon) {
				outRad = outRad + trace(Ray(hit.position + N * epsilon, outDir), depth + 1) * hit.material->kd / M_PI * cosThetaL / pdf / diffuseSelectProb;
			}
		}
		else if (rnd() < diffuseSelectProb + mirrorSelectProb) { // mirror
			double pdf = SampleMirror(N, ray.dir, outDir);
			outRad = outRad + trace(Ray(hit.position + N * epsilon, outDir), depth + 1) * hit.material->ks / pdf / mirrorSelectProb;
		}
		return outRad;
	}*/

	void Animate(float dt) {
		camera.Animate(dt);
		for (Intersectable* object : objects) {
			//object->Animate(dt);
		}
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord);
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao = 0, textureId = 0;	// vertex array object id and texture id
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight) {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

		glGenTextures(1, &textureId);
		glBindTexture(GL_TEXTURE_2D, textureId);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	void LoadTexture(std::vector<vec4>& image) {
		glBindTexture(GL_TEXTURE_2D, textureId);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, &image[0]);
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		int location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
		const unsigned int textureUnit = 0;
		if (location > 0) {
			glUniform1i(location, textureUnit);
			glActiveTexture(GL_TEXTURE0 + textureUnit);
			glBindTexture(GL_TEXTURE_2D, textureId);
		}
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight);
	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	fullScreenTexturedQuad->LoadTexture(image);
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 27) {
		exit(0);
	}
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	scene.Animate(0.05f);
	glutPostRedisplay();
}

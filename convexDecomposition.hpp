#pragma once

// This hpp comes from bullet demo's source.
// Thanks.
// https://github.com/kripken/bullet/blob/master/Demos/ConvexDecompositionDemo/ConvexDecompositionDemo.cpp

#include <bullet/btBulletDynamicsCommon.h>
#include <bullet/LinearMath/btQuickprof.h>
#include <bullet/LinearMath/btGeometryUtil.h>
#include <bullet/BulletCollision/CollisionShapes/btShapeHull.h>
#include <bullet/BulletCollision/CollisionDispatch/btCompoundCollisionAlgorithm.h> //for callback
#include <bullet/ConvexDecomposition/ConvexDecomposition.h>
#include <bullet/HACD/hacdHACD.h>

bool MyCompoundChildShapeCallback(
	const btCollisionShape* pShape0,
	const btCollisionShape* pShape1)
{
	return true;
}

bool MyContactCallback(
	btManifoldPoint& cp,
	const btCollisionObjectWrapper* colObj0Wrap,
	int partId0,
	int index0,
	const btCollisionObjectWrapper* colObj1Wrap,
	int partId1,
	int index1)
{
	if (colObj0Wrap->getCollisionObject()->getCollisionShape()->getShapeType() == COMPOUND_SHAPE_PROXYTYPE)
	{
		btCompoundShape* compound = (btCompoundShape*)colObj0Wrap->getCollisionObject()->getCollisionShape();
		btCollisionShape* childShape;
		childShape = compound->getChildShape(index0);
	}

	if (colObj1Wrap->getCollisionObject()->getCollisionShape()->getShapeType() == COMPOUND_SHAPE_PROXYTYPE)
	{
		btCompoundShape* compound = (btCompoundShape*)colObj1Wrap->getCollisionObject()->getCollisionShape();
		btCollisionShape* childShape;
		childShape = compound->getChildShape(index1);
	}

	return true;
}

class MyConvexDecomposition : public ConvexDecomposition::ConvexDecompInterface
{
public:
	virtual void ConvexDecompResult(ConvexDecomposition::ConvexResult &result)
	{
		btTriangleMesh* trimesh = new btTriangleMesh();
		m_trimeshes.push_back(trimesh);

		centroid.setValue(0, 0, 0);

		btAlignedObjectArray<btVector3> vertices;
		for (unsigned int i = 0; i < result.mHullVcount; i++)
		{
			btVector3 vertex(result.mHullVertices[i * 3], result.mHullVertices[i * 3 + 1], result.mHullVertices[i * 3 + 2]);
			centroid += vertex;
		}

		centroid *= 1.f / (float(result.mHullVcount));
		for (unsigned int i = 0; i < result.mHullVcount; i++)
		{
			btVector3 vertex(result.mHullVertices[i * 3], result.mHullVertices[i * 3 + 1], result.mHullVertices[i * 3 + 2]);
			vertex -= centroid;
			vertices.push_back(vertex);
		}

		const unsigned int *src = result.mHullIndices;
		for (unsigned int i = 0; i < result.mHullTcount; i++)
		{
			unsigned int index0 = *src++;
			unsigned int index1 = *src++;
			unsigned int index2 = *src++;

			btVector3 vertex0(result.mHullVertices[index0 * 3], result.mHullVertices[index0 * 3 + 1], result.mHullVertices[index0 * 3 + 2]);
			btVector3 vertex1(result.mHullVertices[index1 * 3], result.mHullVertices[index1 * 3 + 1], result.mHullVertices[index1 * 3 + 2]);
			btVector3 vertex2(result.mHullVertices[index2 * 3], result.mHullVertices[index2 * 3 + 1], result.mHullVertices[index2 * 3 + 2]);

			vertex0 -= centroid;
			vertex1 -= centroid;
			vertex2 -= centroid;

			trimesh->addTriangle(vertex0, vertex1, vertex2);

			index0 += mBaseCount;
			index1 += mBaseCount;
			index2 += mBaseCount;
		}

		btConvexHullShape* convexShape = new btConvexHullShape(&(vertices[0].getX()), vertices.size());
		convexShape->setMargin(0.01f);
		m_convexShapes.push_back(convexShape);
		m_convexCentroids.push_back(centroid);
		m_collisionShapes.push_back(convexShape);
		mBaseCount += result.mHullVcount; // advance the 'base index' counter.
	}

	int mBaseCount = 0;
	int mHullCount = 0;
	btAlignedObjectArray<btConvexHullShape*> m_convexShapes;
	btAlignedObjectArray<btVector3> m_convexCentroids;
	btAlignedObjectArray<btTriangleMesh*> m_trimeshes;
	btAlignedObjectArray<btConvexHullShape*> m_collisionShapes;
	btVector3 centroid;
};

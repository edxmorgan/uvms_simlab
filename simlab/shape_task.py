import numpy as np


class ShapeFormationController():
    def __init__(self, alphas:np.ndarray):
        self.alphas = alphas

    def get_control_input(self, states:np.ndarray, waypoints:np.ndarray):
        """
        Returns the control input to maintain formation.

        params:
            states -- current states of the system. Size: (3,N)\n
            waypoints -- waypoints of the system. Size: (3,N)\n
        """
        
        n = len(states[0,:])
        alphas = self.alphas
        input = np.zeros((3,1))
        
        q1 = states[:,0]
        q2 = states[:,1]
        q3 = states[:,2]

        d1 = waypoints[:,0]
        d2 = waypoints[:,1]
        d3 = waypoints[:,2]

        sigma21 = np.linalg.norm(q2-q1) - np.linalg.norm(d2-d1)
        sigma31 = np.linalg.norm(q3-q1) - np.linalg.norm(d3-d1)
        sigma32 = np.linalg.norm(q3-q2) - np.linalg.norm(d3-d2)

        u2 = (-alphas[:,1]*(q2-q1)*sigma21).reshape((-1,1))
        u3 = (-alphas[:,2]*((q3-q1)*sigma31 + np.cross(q3-q1,q3-q2)*sigma32)).reshape(-1,1)

        input = np.concatenate((input,u2,u3),axis=1)

        for l in range(3,n):
            i,j,k = np.array([l]*3) - np.array([3,2,1])
            qi = states[:,i]
            qj = states[:,j]
            qk = states[:,k]
            ql = states[:,l]

            di = waypoints[:,i]
            dj = waypoints[:,j]
            dk = waypoints[:,k]
            dl = waypoints[:,l]

            sigmali = np.linalg.norm(ql-qi) - np.linalg.norm(dl-di)
            sigmalj = np.linalg.norm(ql-qj) - np.linalg.norm(dl-dj)
            sigmalk = np.linalg.norm(ql-qk) - np.linalg.norm(dl-dk)

            ul = (-alphas[:,l]*(ql-qi)*sigmali + (ql-qj)*sigmalj + np.cross(ql-qi,ql-qj)*sigmalk).reshape(-1,1)
            
            input = np.concatenate([input, ul],axis=1)
        return input
    

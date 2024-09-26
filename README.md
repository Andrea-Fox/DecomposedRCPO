# Optimal Flow Admission Control in Edge Computing via Safe Reinforcement Learning

With the uptake of intelligent data-driven applications, edge computing infrastructures necessitate a new generation of admission control algorithms to maximize system performance under limited and highly heterogeneous resources. In this paper, we study how to optimally select information flows which belong to different classes and dispatch them to multiple edge servers where applications perform flow analytic tasks. The optimal policy is obtained via the theory of constrained Markov decision processes (CMDP) to take into account the demand of each edge application for specific classes of flows, the constraints on computing capacity of edge servers and the constraints on access network capacity. 

We develop DRCPO, a specialized primal-dual Safe Reinforcement Learning (SRL) method which solves the resulting optimal admission control problem by reward decomposition. DRCPO operates optimal decentralized control and mitigates effectively state-space explosion while preserving optimality. Compared to existing Deep Reinforcement Learning (DRL) solutions, extensive results show that it achieves $15$\% higher reward on a wide variety of environments, while requiring on average only $50$\% learning episodes to converge. Finally, we further improve the system performance by matching DRCPO with load-balancing in order to dispatch optimally information flows to the available edge servers.




The work will be included in the proceedings of WiOpt 2024.

#include <MT/array.h>
#include <MT/opengl.h>

struct MDP{
  arr Ps;    //!< start distribution
  arr Psas;  //!< transition probs
  arr Ras;   //!< reward expectation as a function of (action,x_before)
  double gamma; //!< discounting factor
};

void loadMdpFile(MDP& mdp, const char* filename){
  ifstream fil;
  MT::open(fil,filename);
  arr gam;
  mdp.Psas.readTagged(fil,"Psas");
  mdp.Ps.readTagged(fil,"Ps");
  mdp.Ras.readTagged(fil,"Ras");
  gam.readTagged(fil,"gamma");
  mdp.gamma = gam(0);
}

void simulate(uint& state_new, double& reward, uint action, uint state_old,const MDP& mdp){
  uint n=mdp.Ps.N;

  reward = mdp.Ras(action,state_old);

  arr p(n);
  for(uint s=0;s<n;s++) p(s) = mdp.Psas(s,action,state_old); //same as Psas(:,action,state) in matlab
  state_new = SUS(p); //sample from p
}

int main(int argn, char **argv){
  MT::initCmdLine(argn,argv);

  double lambda = 0.5;
  // get lambda from commandline
  
  if(MT::getCmdLineArgument("lambda"))
      lambda = atof(MT::getCmdLineArgument("lambda"));

  //display
  OpenGL gl;
  
  //load the file
  MDP mdp;
  loadMdpFile(mdp, "miniMaze.mdp");
  //cout <<mdp.Psas <<mdp.Ps <<mdp.Ras <<mdp.gamma <<endl;
    
  //initialize with a random walker
  uint S=mdp.Ps.N,A=mdp.Psas.d1;
  arr pi(S,A); //store the policy such that pi[s] is the distribution over actions in s
  pi = 1./A;   //random policy pi(a|s) = 1/A
  arr Q(S,A);  //same with Q function
  arr e(S,A);  // eligibility trace
  Q.setZero(); //initialize Q(s,a)=0
  e.setZero();


  //learning parameters
  double epsilon = .1;
  double alpha = .1;
  double delta = .1;
  
  uint state_start = argmax(mdp.Ps);
  uint state_old, state_new, action;
  double reward;

  rnd.clockSeed();

  state_new = SUS(mdp.Ps); //sample from the start distribution
  ofstream fil("z.rewards");
  for(uint t=0;t<10000;t++){
    state_old = state_new;

    if(rnd.uni()<epsilon){ //with probability epsilon: choose random action
      action=rnd(A);
      //in Q(lambda): set all eligibility traces zero!
      e.setZero();
    }else{
      action = SUS(pi[state_old]); //sample from pi(:|s)
    }

    simulate(state_new, reward, action, state_old, mdp);

    //cout <<"s=" <<state_old  <<" a=" <<action  <<" s'=" <<state_new  <<" r=" <<reward  <<endl;
    fil <<t <<' ' <<reward <<' ' <<max(Q[state_start]) <<endl;
    
    //update the Q(s,a) function
    double maxQ = max(Q[state_new]);
    // increment eligibility trace
    e(state_old,action) += 1.;
    delta = reward + mdp.gamma*maxQ - Q(state_old,action);
    for(uint s=0;s<S;s++)
        for(uint a=0;a<A;a++) {
            Q(s,a) = Q(s,a)+alpha*delta*e(s,a);
            e(s,a) = mdp.gamma*e(s,a);
        }
 //    Q(state_old,action) += alpha * (reward + mdp.gamma * maxQ - Q(state_old,action));
    
    //update the policy to be greedy policy (exploration handled during action selection)
    pi.setZero();
    for(uint s=0;s<S;s++){
      uint max_action = argmax(Q[s] + 1e-10*randn(A,1));
      pi(s,max_action) = 1.;
    }
    
    //display
    if(true){
      byteA img(3,3);
      img=255;
      img.elem(state_new) = 128;
      img.elem(7)=img.elem(4) = 0;
      gl.watchImage(img,false,10);
      MT::wait(.1);
    }
  }
  fil.close();
  gnuplot("plot 'z.rewards' us 1:2 title 'rewards', 'z.rewards' us 1:3 title 'Value(start)'",NULL,true);

  return 0;
}

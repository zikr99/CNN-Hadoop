package convnet;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author zikr
 */

import java.util.List;
import java.util.ArrayList;
import convnet.Net;
import convnet.Vol;

public class Trainer {
    
    public Net net;
    public String method;
    public double learning_rate;
    public double momentum;
    public int batch_size;
    public double l1_decay;
    public double l2_decay;
    public double ro;
    public double eps;
    public int k;
    public List<double[]> gsum;
    public List<double[]> xsum;
    public int iter;
            
    public Trainer(Net pnet, String pmethod, double plearning_rate, double pmomentum, 
            int pbatch_size, double pl1_decay, double pl2_decay, double pro, double peps)
    {
        net = pnet;
        method = pmethod;
        learning_rate = plearning_rate;
        momentum = pmomentum;
        batch_size = pbatch_size;
        l1_decay = pl1_decay;
        l2_decay = pl2_decay;
        ro = pro;
        eps = peps;
        k = 0;
        iter = 0;
        gsum = new ArrayList<double[]>();
        xsum = new ArrayList<double[]>();
    }
    
    public PassInfo pass(Vol s, int r)
    {
	net.forward(s, 1);
	double A = this.net.backward(r);
	double k = 0;
        double d = 0;
	this.k++;
			
	if (this.k%this.batch_size == 0)
	{
            System.out.println("Update params, batch complete: " + this.k);
            
            List<ParamsGrads> e = this.net.getParamsAndGrads();
			
            if (this.gsum.isEmpty() && (!this.method.equals("sgd") || (this.momentum > 0)))
            {
		for (int E = 0; E < e.size(); E++)
		{
                    this.gsum.add(Util.zeros(e.get(E).params.length));

                    if (this.method.equals("adadelta"))
                    {
			this.xsum.add(Util.zeros(e.get(E).params.length));
                    }
                    else
                    {
			this.xsum.add(new double [0]);
                    }
		}
            }
					
            for (int E = 0; E < e.size(); E++)
            {
		ParamsGrads H = e.get(E);
		double[] w = H.params;
		double[] F = H.grads;

		double z = H.l2_decay_mul;
		double I = H.l1_decay_mul;
		double l = this.l2_decay*z;
		double n = this.l1_decay*I;
		int u = w.length;
						
		for (int B = 0; B < u; B++)
		{
                    k += l*w[B]*w[B]/2;
                    d += n*Math.abs(w[B]);

                    double D = n*((w[B] > 0)?1:-1);
                    double o = l*(w[B]);
                    double t = (o + D + F[B])/this.batch_size;
                    double[] m = this.gsum.get(E);
                    double[] C = this.xsum.get(E);
							
                    if (this.method.equals("adagrad"))
                    {
                        m[B] = m[B] + t*t;
			double v = -this.learning_rate/Math.sqrt(m[B] + this.eps)*t;
			w[B] += v;
                    }
                    else if (this.method.equals("windowgrad"))
                    {
                        m[B] = this.ro*m[B] + (1 - this.ro)*t*t;
                        double v = -this.learning_rate/Math.sqrt(m[B] + this.eps)*t;
									
                        w[B] += v;
                    }
                    else if (this.method.equals("adadelta"))
                    {
                        m[B] = this.ro*m[B] + (1 - this.ro)*t*t;
                        double v = -Math.sqrt((C[B] + this.eps)/(m[B] + this.eps))*t;
                        C[B] = this.ro*C[B] + (1 - this.ro)*v*v;

			w[B] += v;
                    }
                    else if (this.method.equals("nesterov"))
                    {
                        double v = m[B];
                        m[B] = m[B]*this.momentum + this.learning_rate*t;
                        v = this.momentum*v - (1 + this.momentum)*m[B];

                        w[B] += v;
                    }
                    else
                    {
                        if (this.momentum > 0)
                        {
                            double v = this.momentum*m[B] - this.learning_rate*t;
                            m[B] = v;

                            w[B] += v;
                        }
                        else
                        {
                            w[B] += -this.learning_rate*t;
                        }
                    }
                					
                    F[B] = 0;
                }
            }
        }
        
        PassInfo p = new PassInfo();
        
        p.l1_decay_loss = d;
        p.l2_decay_loss = k;
        p.cost_loss = A;
        p.softmax_loss = A;
        p.loss = A + d + k;
        
        return p;
    }
    
    public double evalValErrors(List<Vol> valdata, double[] vallabels)
    {        
	double w = 0;
				
	for (int m = 0; m < valdata.size(); m++)
	{
            Vol u = valdata.get(m);
            double o = vallabels[m];
					
            net.forward(u, (int)o);
					
            int n = net.getPrediction();
            w += ((n == (int)o)?1:0);
	}
				
	w /= valdata.size();			
	return w;
    }
    
    public void step(List<Vol> data, double[] labels, List<Vol> valdata, double[] vallabels,
            int num_epochs)
    {
        this.iter++;
        
        int p = Util.randi(0, data.size());
	Vol u = data.get(p);
	double o = labels[p];

	this.pass(u, (int)o);
			
	int n = num_epochs*batch_size;

        if (this.iter >= n)
        {
            double m = this.evalValErrors(valdata, vallabels);
        
            this.iter = 0;
	}
    }
    
    public void train(List<Vol> data, double[] labels, List<Vol> valdata, double[] vallabels,
            int num_epochs)
    {
        int n = num_epochs*batch_size;
        
        for (int i = 0; i < n; i++)
        {
            step(data, labels, valdata, vallabels, num_epochs);
        }
    }
}

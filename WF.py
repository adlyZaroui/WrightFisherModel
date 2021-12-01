#Projet Processus stochastiques M1IM 2020-2021
#Adrien Krifa, Antonio Soggia, Adly Zaroui
#fork de https://github.com/SasankYadati/wright-fisher-population-genetics-simulation/blob/master/wright_fisher-hints%20(1).md
#Modele de Wright-Fisher

import numpy as np
from random import randint
import matplotlib.pyplot as plt

#DEFINITIONS
#Un individu est modelisé par un booléen '1' ou '0' selon s'il possède respectivement l'allele A ou B
#Une population est un np.array constitué d'individus

#HYPHOTHESES
#La population est haploïde : tous les individus de la population possèdent un et un seul allèle (A ou B) du gène consideré
#La taille de la population est constante au cours du temps
#les generations ne se chevauchent pas
#A chacun des individus de la population au temps n+1 est associé un unique parent (dont il herite de l'allele), uniformement parmi tous les individus de la population au temps n et independamment des autres
#la reproduction à l'instant k ne dépend pas des reproductions précedentes


def population(N,p0):
        ''' prend entier N>=1 et un reel 0<=p0<=1,
            retourne une population de taille N dont
            la proportion d'allèles A est p0 '''

        population = np.zeros(N) #np.array de taille N dont toutes les composantes sont nulles
        population[0:int(p0*N)] = 1 #on rend les int(p0*N) premieres composantes égales à '1'
        np.random.shuffle(population) #permutation uniforme de la population
        return population


def generation(population):
        ''' prend une population,
            retourne un vecteur population de la generation suivante
            suivant le modele de Wright-Fisher '''

        N = len(population)
        return np.random.choice(population, N, replace = True) #choisit uniformement N individus de population avec remise

def X(n,population_initiale):
        ''' prend entier n>=1, une population initiale,
            retourne la liste [X0,X1,X2,...Xn] ''' #où Xi est le nombre d'allèles A dans la population au temps i

        X = [int(population_initiale.sum())] #initialisation de X = [X0]
        population = population_initiale

        for i in range(n):
                population = generation(population) #la population passe à la generation suivante
                X.append(int(population.sum())) #compte le nombre de '1' dans la generation courante et le rajoute à la fin de la liste X
        return X

#def Wright_Fisher(N,p0,n):
#        p=population(N,p0)
#        GEN=np.zeros((n,N))
#        GEN[0]=p #population initiale
#        for i in range(1,n-1):
#            p1=generation(p)
#            GEN[i]=p1
#        print("Population initiale: "+str(p)+"\n"+"Nombres d'individus A à la génération suivante: "+str(GEN[1])+"\n")
#        print("Répartition de la population à chaque génération: "+"\n"+str(GEN)+"\n"+"\n"+"Effectifs à chaque génération: ")
#        plot de l'histogramme correspondant au nombre de A au fil des générations
#        plt.title("Modèle de Wright-Fisher, population de taille {}, sur {} générations".format(N,n))
#        plt.xlabel("Générations")
#        plt.ylabel("Nombre d'allèles A")
#        plt.plot(X)
#        plt.grid()
#        plt.show()
#        return X(n,p)



def T(X,A):
        ''' prend X une liste generee par la fonction X,
            A une partie de l'ensemble {0,...,N}
            et retourne le temps d'atteinte de A, inf{n € IN*, Xn € A} '''
        N=len(X)
        for i in range(N):
                if X[i] in A:
                        return i
        print('A '+'non atteint')

#X=Wright_Fisher(20,0.4,30)

#plot de l'histogramme correspondant au nombre de A au fil des générations
#plt.title("Modèle de Wright-Fisher, population de taille {}, sur {} générations".format(N,n))
#plt.xlabel("Générations")
#plt.ylabel("Nombre d'allèles A")
#plt.plot(X)
#plt.grid()
#plt.show()

#X = X(10000,population(100,0.4)) #Il y a un problème ici, rien de grave, je compte sur toi pour le résoudre
#T(X,{0,len(X)})

### Avec ce code, la commande X(n,population(N,p0)) renvoi la liste [X0, ..., Xn] du nombre d'allèles A des n+1 premieres generations
### avec une population initiale de N individus,dont la proportion d'allèles A est de p0.



########################################################################################################################
#######################   ILLUSTRATION PAR LA SIMULATION DES RESULTATS THEORIQUES  #####################################
########################################################################################################################

# Corollaire 2.3: la probabilité sachant X0 = i, que X atteigne N (avant 0) est égale à i/N



def simulation(N):
        ''' prend N € IN et cree des generations jusqu'à fixation
            renvoi le temps d'atteinte de {N} s'il est fini, -1 sinon
            (dans ce cas {0} a été atteint avant)
            Attention cette fonction ne sert que dans la partie illustration,
            autrement, ne pas en tenir compte  '''

        i0 = i = randint(0,N) #X0
        population_initiale = pop = population(N, i/N)
        T = 0 #compteur
        while i > 0 and i < N:
                pop = generation(pop)
                i = pop.sum() #Xi
                T = T + 1
        if i == N: return (T,i) #dans ce cas, {N} a été atteint avant {0}, et T est le temps d'atteinte
        return (0,i) #dans ce cas {0} a été atteint avant {N}


#### ce qui suit ne devra être executé qu'une fois SVP

L_N = []
T_i = [] #liste des N et des (T,i)
for i in range(1000):
        L_N.append(randint(100,1000)) #on tire des N au hasard et on les ajoutes à la liste des N
        T_i.append(simulation(L_N[i])) #pour chaque N on applique une simulation et on rajoute la sortie (T,i) à la liste des (T,i)


#les (i, i/N) seront donnés par (T_i[k][1] , T_i[k][1]/L_N[k])
#les (i,T/N) seront donnés par  (T_i[k][1] , T_i[k][0]/L_N[k]) pour 0 <= k <= 999
#reste a afficher les 2 nuages de points ci-après
#   (i,i/N) proba theorique de fixation
#   (i,T) moyenne empirique de fixation

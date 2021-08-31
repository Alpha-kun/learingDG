using FastGaussQuadrature
using LinearAlgebra
using Plots

#Domain: [-1,1]

order=4
N=32
h=1/N #half of element size

#create quadrature points
nodes, weights = gausslegendre(order)

#compute interpolation vector
Lₗ=zeros(order)
Lᵣ=zeros(order)
for i in 1:order
    denom = prod(nodes[i] .- nodes[filter(x -> x != i, 1:order)])
    Lₗ[i]=prod(-1 .- nodes[filter(x -> x != i, 1:order)]) / denom
    Lᵣ[i]=prod(1 .- nodes[filter(x -> x != i, 1:order)]) / denom
end

#compute mass matrix
M=h*Diagonal(weights)

#compute stiffness matrix
K=zeros(order,order)
for i in 1:order
    denom = prod(nodes[i] .- nodes[filter(x -> x != i, 1:order)])
    for j in 1:order
        if i==j
            L̇ᵢ=sum(1 ./ (nodes[i] .- nodes[filter(x -> x != i, 1:order)]))
            K[i,j]=weights[j]*L̇ᵢ
        else
            L̇ᵢ=prod(nodes[j] .- nodes[filter(x -> (x != i) & (x != j), 1:order)]) / denom
            K[i,j]=weights[j]*L̇ᵢ
        end
    end
end

#########################################################
################ problem specific set up ################
#########################################################

#initial condition
function ψ(x)
    return -sin.(pi*x)
end

function f(x)
    return x.^2/2
end

#Lax-Friedrich flux for invicid burgers

function fs(u⁻,u⁺)
    #if ((u⁺+u⁻)/2) > 0
    #    return u⁻^2/2
    #else
    #    return u⁺^2/2
    #end

    return 0.5*(f(u⁻)+f(u⁺))-0.5*max(abs(u⁻),abs(u⁺))*(u⁺-u⁻)

    #if (u⁻>0) & (u⁺>0)
    #    return u⁻^2/2
    #elseif (u⁻<0) & (u⁺<0)
    #    return u⁺^2/2
    #else
    #    return (abs(u⁻)<=abs(u⁺) ? u⁺^2/2 : u⁻^2/2)
    #end

    #if (f(u⁺)-f(u⁻)/(u⁺-u⁻)) > 0
    #    return u⁻^2/2
    #else
    #    return u⁺^2/2
    #end
end

function flux_term(U)
    f=0*U
    for i in 1:N
        u⁻ = (i==1 ? U[:,N] : U[:,i-1])
        u⁺ = (i==N ? U[:,1] : U[:,i+1])
        f⁻=fs(dot(u⁻, Lᵣ), dot(U[:,i], Lₗ))*Lₗ
        f⁺=fs(dot(U[:,i], Lᵣ), dot(u⁺, Lₗ))*Lᵣ
        f[:,i]=f⁺-f⁻
    end
    return f
end

#set up nodes
x=reduce(hcat, [(-1+(2*i-1)*h) .+ h*nodes for i in 1:N])
#set up initial nodal values
u=ψ(x)

t=0
T=1.5
dt=0.005

anime = @animate for i in 1:500
    k1=M\(K*f(u)-flux_term(u))
    k2=M\(K*(f(u+0.5dt*k1))-flux_term(u+0.5dt*k1))
    k3=M\(K*(f(u+0.5dt*k2))-flux_term(u+0.5dt*k2))
    k4=M\(K*(f(u+dt*k3))-flux_term(u+dt*k3))
    u=u+dt*(k1+2k2+2k3+k4)/6
    plot(x,u,ylims=(-2.6,2.6),legend=false)
    p=scatter!(x,u,legend=false)
    display(p)
    #sleep(0.1)
    t+=dt
end

gif(anime, "D:\\side_projects\\DG\\DG_burgers_discontinuous_2.gif", fps = 30)
#plot(vec(x'),vec(a'))

plot(x,u,legend=false)
scatter!(x,u,legend=false)

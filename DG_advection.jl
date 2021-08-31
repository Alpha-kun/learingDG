using FastGaussQuadrature
using LinearAlgebra
using Plots

#Domain: [-1,1]

order=6
N=8 #number of element
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
    return ((-0.25 .< x) .& (x .< 0.25)).*(0.25 .- abs.(x)) #sin.(pi*x)
end

#upwind flux for advection with c>0
function fs(u⁻,u⁺)
    return u⁻
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
T=2
dt=0.01

#rk4 time stepper
anime = @animate for i in 1:200
    k1=M\(K*u-flux_term(u))
    k2=M\(K*(u+0.5dt*k1)-flux_term(u+0.5dt*k1))
    k3=M\(K*(u+0.5dt*k2)-flux_term(u+0.5dt*k2))
    k4=M\(K*(u+dt*k3)-flux_term(u+dt*k3))
    u=u+dt*(k1+2k2+2k3+k4)/6
    plot(x,u,ylims=(-0.1,0.4),legend=false)
    p=scatter!(x,u,legend=false)
    display(p)
    #sleep(0.1)
    t+=dt
end

gif(anime, "D:\\side_projects\\DG\\DG_advection_discontinuous.gif", fps = 15)
#plot(vec(x'),vec(a'))

plot(x,u,legend=false)
scatter!(x,u,legend=false)

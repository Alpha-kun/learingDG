using FastGaussQuadrature
using LinearAlgebra
using Plots

#Domain: [-1,1]

order=4
N=32
h=1/N #half of element size

#create quadrature points
nodes, weights = gausslobatto(order)

#compute mass matrix
M=h*Diagonal(weights)

#compute derivative matrix
D=zeros(order,order)
for i in 1:order
    denom = prod(nodes[i] .- nodes[filter(x -> x != i, 1:order)])
    for j in 1:order
        if i==j
            L̇ᵢ=sum(1 ./ (nodes[i] .- nodes[filter(x -> x != i, 1:order)]))
            D[i,j]=L̇ᵢ
        else
            L̇ᵢ=prod(nodes[j] .- nodes[filter(x -> (x != i) & (x != j), 1:order)]) / denom
            D[i,j]=L̇ᵢ
        end
    end
end
D=(1/h)*D'

#########################################################
################ problem specific set up ################
#########################################################

#initial condition
function ψ(x)
    return -sin.(pi*x).+1.0
    #return abs.(x)
    #return ((-1.0 .< x) .& (x .< 0.0)).*(-1 .- x) + ((0.0 .<= x) .& (x .< 1.0)).*(1 .-x)
    #return ((-1.0 .< x) .& (x .< -0.25)).*(1) + ((0.25 .<= x) .& (x .< 1.0)).*(-1)
end

function f(x)
    return x.^2/2
end

#Lax-Friedrich flux for invicid burgers
function fs(u⁻,u⁺)
    #return 0.5*(f(u⁻)+f(u⁺))-(1/12)*(u⁺-u⁻)^2
    return 0.5*(f(u⁻)+f(u⁺))-0.5*max(abs(u⁺),abs(u⁻))*(u⁺-u⁻)
end

function flux_term(U)
    f=0*U
    for i in 1:N
        u⁻ = (i==1 ? U[:,N] : U[:, i-1])
        u⁺ = (i==N ? U[:,1] : U[:, i+1])
        f⁻=fs(u⁻[order], U[1,i])
        f⁺=fs(U[order,i], u⁺[1])
        f[1,i]=-f⁻
        f[order,i]=f⁺
    end
    return f
end

#set up nodes
x=reduce(hcat, [(-1+(2*i-1)*h) .+ h*nodes for i in 1:N])
#set up initial nodal values
u=ψ(x)

#Gassner's split operator formulation
α=2/3
function F(u)
    volume_1 = α*(M*D)'*f(u)
    volume_2 = (1-α)*(M*u).*(D*0.5u)
    volume_3 = (1-α)*(D'*M*u).*(0.5u)
    return M\(volume_1 - volume_2 + volume_3 - flux_term(u))
end

t=0
T=1.5
dt=0.005

anime = @animate for i in 1:1000
    k1=F(u)
    k2=F(u+0.5dt*k1)
    k3=F(u+0.5dt*k2)
    k4=F(u+dt*k3)
    u=u+dt*(k1+2k2+2k3+k4)/6
    plot(x,u,ylims=(-2.6,2.6),legend=false)
    p=scatter!(x,u,legend=false)
    display(p)
    #sleep(0.5)
    t+=dt
end

gif(anime, "D:\\side_projects\\DG\\DG_burgers_2antishock.gif", fps = 30)
#plot(vec(x'),vec(a'))

plot(x,u,legend=false)
scatter!(x,u,legend=false)

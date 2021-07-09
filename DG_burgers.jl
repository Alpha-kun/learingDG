using FastGaussQuadrature
using LinearAlgebra
using Plots

#Domain: [-1,1]

order=4
N=16
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
    return sin.(pi*x)
end

#upwind flux for advection with c>0
function fs(u⁻,u⁺)
    return (u⁻<u⁺ ? u⁺ : u⁻)^2/2
end

function flux_term(a)
    f=0*a
    for i in 1:N
        a⁻ = (i==1 ? a[N,:] : a[i-1, :])
        a⁺ = (i==N ? a[1,:] : a[i+1, :])
        f⁻=fs(dot(a⁻, Lᵣ), dot(a[i,:], Lₗ))*Lₗ
        f⁺=fs(dot(a[i,:], Lᵣ), dot(a⁺, Lₗ))*Lᵣ
        f[i,:]=f⁺-f⁻
    end
    return f
end

function f(x)
    return x.^2/2
end

x=zeros(N,order)
a=zeros(N,order)
for i in 1:N
    x[i,:] = (-1+(2*i-1)*h) .+ h*nodes
    a[i,:] = ψ(x[i,:])
end



t=0
T=2
dt=0.001

while t<T
    k1=M\(K*f(a)'-flux_term(a)')
    k2=M\(K*(f(a)'+0.5dt*k1)-flux_term(a+0.5dt*k1')')
    k3=M\(K*(f(a)'+0.5dt*k2)-flux_term(a+0.5dt*k2')')
    k4=M\(K*(f(a)'+dt*k3)-flux_term(a+dt*k3')')
    a=a+dt*(k1'+2k2'+2k3'+k4')/6
    plot(x',a',legend=false)
    p=scatter!(x',a',legend=false)
    display(p)
    sleep(0.2)
    t+=dt
end

#plot(vec(x'),vec(a'))

plot(x',a',legend=false)
scatter!(x',a',legend=false)

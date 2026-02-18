'use client';

import Image from "next/image";
import { useRouter } from "next/navigation";
import { useState } from 'react';
import { zodResolver } from '@hookform/resolvers/zod';
import { useForm } from 'react-hook-form';
import * as z from 'zod';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Progress } from '@/components/ui/progress';
import { Loader2, AlertCircle, ArrowLeft } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

const formSchema = z.object({
  gender: z.enum(['Male', 'Female']),
  SeniorCitizen: z.enum(['0', '1']),
  Partner: z.enum(['Yes', 'No']),
  Dependents: z.enum(['Yes', 'No']),
  tenure: z.string().min(1, 'Tenure is required').max(2, 'Max 2 digits'),
  PhoneService: z.enum(['Yes', 'No']),
  MultipleLines: z.enum(['Yes', 'No']),
  InternetService: z.enum(['Fiber optic', 'DSL', 'No']),
  OnlineSecurity: z.enum(['Yes', 'No']),
  OnlineBackup: z.enum(['Yes', 'No']),
  DeviceProtection: z.enum(['Yes', 'No']),
  TechSupport: z.enum(['Yes', 'No']),
  StreamingTV: z.enum(['Yes', 'No']),
  StreamingMovies: z.enum(['Yes', 'No']),
  Contract: z.enum(['Month-to-month', 'One year', 'Two year']),
  PaperlessBilling: z.enum(['Yes', 'No']),
  PaymentMethod: z.enum([
    'Electronic check',
    'Mailed check',
    'Bank transfer (automatic)',
    'Credit card (automatic)',
  ]),
  MonthlyCharges: z.string().min(1, 'Monthly Charges is required'),
  TotalCharges: z.string().min(1, 'Total Charges is required'),
});

type FormValues = z.infer<typeof formSchema>;

export default function PredictPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const form = useForm<FormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      gender: 'Male',
      SeniorCitizen: '0',
      Partner: 'No',
      Dependents: 'No',
      tenure: '2',
      PhoneService: 'Yes',
      MultipleLines: 'Yes',
      InternetService: 'Fiber optic',
      OnlineSecurity: 'No',
      OnlineBackup: 'No',
      DeviceProtection: 'No',
      TechSupport: 'No',
      StreamingTV: 'Yes',
      StreamingMovies: 'Yes',
      Contract: 'Month-to-month',
      PaperlessBilling: 'Yes',
      PaymentMethod: 'Electronic check',
      MonthlyCharges: '110',
      TotalCharges: '220',
    },
  });

  async function onSubmit(values: FormValues) {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
      const res = await fetch(`${apiUrl}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams(values as any),
      });

      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`Server error: ${res.status} - ${errorText}`);
      }

      const data = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Failed to get prediction. Is the backend running?');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-[#070714] text-[#E8E8F0] overflow-x-hidden relative">

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;0,9..40,800;0,9..40,900&display=swap');
        * { font-family: 'DM Sans', sans-serif; }

        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(12px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes glowPulse {
          0%,100% { box-shadow: 0 0 24px rgba(0,255,178,0.3); }
          50%      { box-shadow: 0 0 52px rgba(0,255,178,0.6); }
        }
        @keyframes scanline {
          from { top: -10%; } to { top: 110%; }
        }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
        @keyframes spin { to { transform: rotate(360deg); } }

        .glow-btn { animation: glowPulse 2.4s ease infinite; }
        .fade-in  { animation: fadeIn .5s ease both; }
        .scan     {
          position: absolute; left: 0; right: 0; height: 40px;
          background: linear-gradient(transparent, rgba(0,255,178,0.02), transparent);
          animation: scanline 5s linear infinite; pointer-events: none;
        }

        ::-webkit-scrollbar       { width: 5px; }
        ::-webkit-scrollbar-track { background: #070714; }
        ::-webkit-scrollbar-thumb { background: #1a1a2e; border-radius: 3px; }
        ::selection               { background: rgba(0,255,178,0.25); }

        /* Custom form styles to match dark theme */
        .custom-card {
          background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
          border: 1px solid rgba(255,255,255,0.08);
          backdrop-filter: blur(10px);
        }
        .section-divider {
          border-bottom: 1px solid rgba(0,255,178,0.15);
        }
      `}</style>

      {/* Grid bg */}
      <div
        className="fixed inset-0 z-0 pointer-events-none opacity-[0.045]"
        style={{
          backgroundImage:
            "linear-gradient(rgba(0,255,178,0.6) 1px, transparent 1px), linear-gradient(90deg, rgba(0,255,178,0.6) 1px, transparent 1px)",
          backgroundSize: "56px 56px",
        }}
      />

      {/* ══════ NAV ══════ */}
      <nav className="fixed top-0 left-0 right-0 z-50 h-16 px-6 md:px-12 flex items-center justify-between bg-[rgba(7,7,20,0.82)] backdrop-blur-xl border-b border-white/[0.05]">
        <div className="flex items-center gap-2.5">
          <div className="relative w-8 h-8 flex-shrink-0">
            <Image
              src="/StayFlow.png"
              alt="StayFlow logo"
              fill
              className="object-contain rounded-lg"
              priority
            />
          </div>
          <span className="font-bold text-[17px] tracking-tight text-white">
            StayFlow<span className="text-[#00FFB2]">.</span>
          </span>
        </div>

        <button
          onClick={() => router.push("/")}
          className="flex items-center gap-2 text-[#00FFB2] hover:text-[#00FFB2]/80 transition-colors text-[13px] font-semibold"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Home
        </button>
      </nav>

      {/* ══════ CONTENT ══════ */}
      <div className="relative z-10 pt-24 pb-16 px-6 md:px-12">
        <div className="max-w-5xl mx-auto">

          {/* Header */}
          <div className="text-center mb-12 fade-in">
            <div className="inline-flex items-center gap-2 bg-[rgba(0,255,178,0.08)] border border-[rgba(0,255,178,0.2)] rounded-full px-4 py-1.5 mb-6">
              <span className="w-1.5 h-1.5 rounded-full bg-[#00FFB2] shadow-[0_0_8px_#00FFB2]" style={{ animation: 'blink 1.8s ease infinite' }} />
              <span className="text-[11px] text-[#00FFB2] font-bold tracking-widest">19 FEATURES · FULL MODEL</span>
            </div>
            <h1 className="text-[clamp(32px,5vw,56px)] font-black leading-[1.04] tracking-[-2px] text-white mb-4">
              Customer Churn<br className="hidden sm:block" />
              <span className="bg-gradient-to-r from-[#00FFB2] to-[#00BFFF] bg-clip-text text-transparent"> Prediction</span>
            </h1>
            <p className="text-[16px] text-[#666] max-w-[560px] mx-auto">
              Fill out all 19 fields below to get a complete churn risk analysis from the ML model.
            </p>
          </div>

          {/* Form Card */}
          <Card className="custom-card shadow-2xl border-0 mb-12 overflow-hidden relative">
            <div className="scan" />
            <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-[#00FFB2] to-transparent opacity-50" />
            
            <CardHeader className="pb-6 border-b border-white/[0.05]">
              <CardTitle className="text-2xl font-black tracking-tight text-white">Enter Customer Details</CardTitle>
              <CardDescription className="text-[#555]">Provide accurate data for each field to maximize prediction accuracy.</CardDescription>
            </CardHeader>

            <CardContent className="pt-8">
              <Form {...form}>
                <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-10">

                  {/* Section 1: Demographics */}
                  <div className="space-y-6">
                    <h3 className="text-lg font-bold section-divider pb-3 text-[#00FFB2] tracking-wide flex items-center gap-2">
                      <span className="text-[10px] font-mono bg-[rgba(0,255,178,0.12)] px-2 py-1 rounded border border-[rgba(0,255,178,0.3)]">01</span>
                      DEMOGRAPHICS
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                      
                      <FormField control={form.control} name="gender" render={({ field }) => (
                        <FormItem>
                          <FormLabel className="text-[#888] text-xs font-semibold tracking-wider">Gender</FormLabel>
                          <Select onValueChange={field.onChange} defaultValue={field.value}>
                            <FormControl>
                              <SelectTrigger className="bg-white/[0.04] border-white/10 text-[#E8E8F0] h-11 rounded-lg">
                                <SelectValue placeholder="Select" />
                              </SelectTrigger>
                            </FormControl>
                            <SelectContent className="bg-[#0a0f20] border-white/10 text-[#E8E8F0]">
                              <SelectItem value="Male">Male</SelectItem>
                              <SelectItem value="Female">Female</SelectItem>
                            </SelectContent>
                          </Select>
                          <FormMessage />
                        </FormItem>
                      )} />

                      <FormField control={form.control} name="SeniorCitizen" render={({ field }) => (
                        <FormItem>
                          <FormLabel className="text-[#888] text-xs font-semibold tracking-wider">Senior Citizen</FormLabel>
                          <Select onValueChange={field.onChange} defaultValue={field.value}>
                            <FormControl>
                              <SelectTrigger className="bg-white/[0.04] border-white/10 text-[#E8E8F0] h-11 rounded-lg">
                                <SelectValue placeholder="Select" />
                              </SelectTrigger>
                            </FormControl>
                            <SelectContent className="bg-[#0a0f20] border-white/10 text-[#E8E8F0]">
                              <SelectItem value="0">No</SelectItem>
                              <SelectItem value="1">Yes</SelectItem>
                            </SelectContent>
                          </Select>
                          <FormMessage />
                        </FormItem>
                      )} />

                      <FormField control={form.control} name="Partner" render={({ field }) => (
                        <FormItem>
                          <FormLabel className="text-[#888] text-xs font-semibold tracking-wider">Has Partner?</FormLabel>
                          <Select onValueChange={field.onChange} defaultValue={field.value}>
                            <FormControl>
                              <SelectTrigger className="bg-white/[0.04] border-white/10 text-[#E8E8F0] h-11 rounded-lg">
                                <SelectValue placeholder="Select" />
                              </SelectTrigger>
                            </FormControl>
                            <SelectContent className="bg-[#0a0f20] border-white/10 text-[#E8E8F0]">
                              <SelectItem value="Yes">Yes</SelectItem>
                              <SelectItem value="No">No</SelectItem>
                            </SelectContent>
                          </Select>
                          <FormMessage />
                        </FormItem>
                      )} />

                      <FormField control={form.control} name="Dependents" render={({ field }) => (
                        <FormItem>
                          <FormLabel className="text-[#888] text-xs font-semibold tracking-wider">Has Dependents?</FormLabel>
                          <Select onValueChange={field.onChange} defaultValue={field.value}>
                            <FormControl>
                              <SelectTrigger className="bg-white/[0.04] border-white/10 text-[#E8E8F0] h-11 rounded-lg">
                                <SelectValue placeholder="Select" />
                              </SelectTrigger>
                            </FormControl>
                            <SelectContent className="bg-[#0a0f20] border-white/10 text-[#E8E8F0]">
                              <SelectItem value="Yes">Yes</SelectItem>
                              <SelectItem value="No">No</SelectItem>
                            </SelectContent>
                          </Select>
                          <FormMessage />
                        </FormItem>
                      )} />

                      <FormField control={form.control} name="tenure" render={({ field }) => (
                        <FormItem>
                          <FormLabel className="text-[#888] text-xs font-semibold tracking-wider">Tenure (months)</FormLabel>
                          <FormControl>
                            <Input
                              type="number"
                              placeholder="0–72"
                              min="0"
                              max="72"
                              className="bg-white/[0.04] border-white/10 text-[#E8E8F0] h-11 rounded-lg"
                              {...field}
                            />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )} />

                    </div>
                  </div>

                  {/* Section 2: Services */}
                  <div className="space-y-6">
                    <h3 className="text-lg font-bold section-divider pb-3 text-[#00FFB2] tracking-wide flex items-center gap-2">
                      <span className="text-[10px] font-mono bg-[rgba(0,255,178,0.12)] px-2 py-1 rounded border border-[rgba(0,255,178,0.3)]">02</span>
                      SERVICES
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                      
                      {['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'].map((fieldName) => (
                        <FormField
                          key={fieldName}
                          control={form.control}
                          name={fieldName as any}
                          render={({ field }) => (
                            <FormItem>
                              <FormLabel className="text-[#888] text-xs font-semibold tracking-wider">
                                {fieldName.replace(/([A-Z])/g, ' $1').trim()}
                              </FormLabel>
                              <Select onValueChange={field.onChange} defaultValue={field.value}>
                                <FormControl>
                                  <SelectTrigger className="bg-white/[0.04] border-white/10 text-[#E8E8F0] h-11 rounded-lg">
                                    <SelectValue placeholder="Select" />
                                  </SelectTrigger>
                                </FormControl>
                                <SelectContent className="bg-[#0a0f20] border-white/10 text-[#E8E8F0]">
                                  <SelectItem value="Yes">Yes</SelectItem>
                                  <SelectItem value="No">No</SelectItem>
                                </SelectContent>
                              </Select>
                              <FormMessage />
                            </FormItem>
                          )}
                        />
                      ))}

                      <FormField control={form.control} name="InternetService" render={({ field }) => (
                        <FormItem>
                          <FormLabel className="text-[#888] text-xs font-semibold tracking-wider">Internet Service</FormLabel>
                          <Select onValueChange={field.onChange} defaultValue={field.value}>
                            <FormControl>
                              <SelectTrigger className="bg-white/[0.04] border-white/10 text-[#E8E8F0] h-11 rounded-lg">
                                <SelectValue placeholder="Select" />
                              </SelectTrigger>
                            </FormControl>
                            <SelectContent className="bg-[#0a0f20] border-white/10 text-[#E8E8F0]">
                              <SelectItem value="Fiber optic">Fiber optic</SelectItem>
                              <SelectItem value="DSL">DSL</SelectItem>
                              <SelectItem value="No">No</SelectItem>
                            </SelectContent>
                          </Select>
                          <FormMessage />
                        </FormItem>
                      )} />

                    </div>
                  </div>

                  {/* Section 3: Contract & Billing */}
                  <div className="space-y-6">
                    <h3 className="text-lg font-bold section-divider pb-3 text-[#00FFB2] tracking-wide flex items-center gap-2">
                      <span className="text-[10px] font-mono bg-[rgba(0,255,178,0.12)] px-2 py-1 rounded border border-[rgba(0,255,178,0.3)]">03</span>
                      CONTRACT & BILLING
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                      
                      <FormField control={form.control} name="Contract" render={({ field }) => (
                        <FormItem>
                          <FormLabel className="text-[#888] text-xs font-semibold tracking-wider">Contract Type</FormLabel>
                          <Select onValueChange={field.onChange} defaultValue={field.value}>
                            <FormControl>
                              <SelectTrigger className="bg-white/[0.04] border-white/10 text-[#E8E8F0] h-11 rounded-lg">
                                <SelectValue placeholder="Select" />
                              </SelectTrigger>
                            </FormControl>
                            <SelectContent className="bg-[#0a0f20] border-white/10 text-[#E8E8F0]">
                              <SelectItem value="Month-to-month">Month-to-month</SelectItem>
                              <SelectItem value="One year">One year</SelectItem>
                              <SelectItem value="Two year">Two year</SelectItem>
                            </SelectContent>
                          </Select>
                          <FormMessage />
                        </FormItem>
                      )} />

                      <FormField control={form.control} name="PaperlessBilling" render={({ field }) => (
                        <FormItem>
                          <FormLabel className="text-[#888] text-xs font-semibold tracking-wider">Paperless Billing</FormLabel>
                          <Select onValueChange={field.onChange} defaultValue={field.value}>
                            <FormControl>
                              <SelectTrigger className="bg-white/[0.04] border-white/10 text-[#E8E8F0] h-11 rounded-lg">
                                <SelectValue placeholder="Select" />
                              </SelectTrigger>
                            </FormControl>
                            <SelectContent className="bg-[#0a0f20] border-white/10 text-[#E8E8F0]">
                              <SelectItem value="Yes">Yes</SelectItem>
                              <SelectItem value="No">No</SelectItem>
                            </SelectContent>
                          </Select>
                          <FormMessage />
                        </FormItem>
                      )} />

                      <FormField control={form.control} name="PaymentMethod" render={({ field }) => (
                        <FormItem>
                          <FormLabel className="text-[#888] text-xs font-semibold tracking-wider">Payment Method</FormLabel>
                          <Select onValueChange={field.onChange} defaultValue={field.value}>
                            <FormControl>
                              <SelectTrigger className="bg-white/[0.04] border-white/10 text-[#E8E8F0] h-11 rounded-lg">
                                <SelectValue placeholder="Select" />
                              </SelectTrigger>
                            </FormControl>
                            <SelectContent className="bg-[#0a0f20] border-white/10 text-[#E8E8F0]">
                              <SelectItem value="Electronic check">Electronic check</SelectItem>
                              <SelectItem value="Mailed check">Mailed check</SelectItem>
                              <SelectItem value="Bank transfer (automatic)">Bank transfer (automatic)</SelectItem>
                              <SelectItem value="Credit card (automatic)">Credit card (automatic)</SelectItem>
                            </SelectContent>
                          </Select>
                          <FormMessage />
                        </FormItem>
                      )} />

                      <FormField control={form.control} name="MonthlyCharges" render={({ field }) => (
                        <FormItem>
                          <FormLabel className="text-[#888] text-xs font-semibold tracking-wider">Monthly Charges ($)</FormLabel>
                          <FormControl>
                            <Input
                              type="number"
                              step="0.01"
                              placeholder="e.g. 70.50"
                              className="bg-white/[0.04] border-white/10 text-[#E8E8F0] h-11 rounded-lg"
                              {...field}
                            />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )} />

                      <FormField control={form.control} name="TotalCharges" render={({ field }) => (
                        <FormItem>
                          <FormLabel className="text-[#888] text-xs font-semibold tracking-wider">Total Charges ($)</FormLabel>
                          <FormControl>
                            <Input
                              type="number"
                              step="0.01"
                              placeholder="e.g. 840.00"
                              className="bg-white/[0.04] border-white/10 text-[#E8E8F0] h-11 rounded-lg"
                              {...field}
                            />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )} />

                    </div>
                  </div>

                  {/* Submit + Reset Buttons */}
                  <div className="flex flex-col sm:flex-row gap-4">
                    <Button
                      type="submit"
                      disabled={loading}
                      className={`glow-btn h-14 text-lg font-black bg-[#00FFB2] text-[#070714] hover:bg-[#00FFB2]/90 rounded-xl ${result ? 'flex-1' : 'w-full'}`}
                    >
                      {loading ? (
                        <>
                          <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                          Analyzing with ML Model...
                        </>
                      ) : (
                        '⚡ Predict Churn Risk'
                      )}
                    </Button>

                    {result && (
                      <Button
                        type="button"
                        onClick={() => {
                          form.reset();
                          setResult(null);
                          setError(null);
                        }}
                        disabled={loading}
                        className="fade-in flex-1 sm:flex-none sm:w-auto h-14 text-lg font-bold bg-transparent border-2 border-[#00FFB2]/30 text-[#00FFB2] hover:bg-[#00FFB2]/10 hover:border-[#00FFB2] rounded-xl transition-all"
                      >
                        ↻ New Prediction
                      </Button>
                    )}
                  </div>

                </form>
              </Form>
            </CardContent>
          </Card>

          {/* Result Card */}
          {result && (
            <Card
              className="custom-card shadow-2xl border-0 overflow-hidden relative fade-in"
              style={{
                borderTop: `3px solid ${result.prediction === 1 ? '#FF5C5C' : '#00FFB2'}`,
              }}
            >
              <div className="scan" />
              <CardHeader className="text-center pb-4 border-b border-white/[0.05]">
                <CardTitle className="text-3xl font-black text-white tracking-tight">Prediction Result</CardTitle>
              </CardHeader>
              <CardContent className="pt-8 space-y-8">
                
                {/* Main result */}
                <div className="text-center">
                  <h2
                    className="text-5xl font-black mb-4 tracking-tight"
                    style={{ color: result.prediction === 1 ? '#FF5C5C' : '#00FFB2' }}
                  >
                    {result.churn}
                  </h2>
                  <p className="text-2xl text-[#888] font-semibold">
                    Churn Probability: <strong className="text-white">{result.churn_probability}%</strong>
                  </p>
                </div>

                {/* Progress bar */}
                <div className="space-y-3">
                  <div className="flex justify-between text-sm font-semibold text-[#555]">
                    <span>0%</span>
                    <span>100%</span>
                  </div>
                  <div className="h-6 rounded-full overflow-hidden bg-white/[0.06] relative">
                    <div
                      className="h-full rounded-full transition-all duration-1000"
                      style={{
                        width: `${result.churn_probability}%`,
                        background: result.churn_probability > 70 ? '#FF5C5C' : result.churn_probability > 40 ? '#FFCC00' : '#00FFB2',
                        boxShadow: `0 0 20px ${result.churn_probability > 70 ? '#FF5C5C' : result.churn_probability > 40 ? '#FFCC00' : '#00FFB2'}88`,
                      }}
                    />
                  </div>
                </div>

                {/* Risk badge */}
                <div className="flex justify-center">
                  <span
                    className="px-12 py-5 rounded-2xl text-white text-2xl font-black shadow-2xl"
                    style={{
                      background: result.risk_level.includes('High') ? '#FF5C5C' : result.risk_level.includes('Medium') ? '#FFCC00' : '#00FFB2',
                      boxShadow: `0 0 30px ${result.risk_level.includes('High') ? '#FF5C5C' : result.risk_level.includes('Medium') ? '#FFCC00' : '#00FFB2'}66`,
                    }}
                  >
                    {result.risk_level}
                  </span>
                </div>

              </CardContent>
            </Card>
          )}

          {/* Error Alert */}
          {error && (
            <Alert variant="destructive" className="mt-10 bg-red-500/10 border-red-500/30 fade-in">
              <AlertCircle className="h-5 w-5 text-red-400" />
              <AlertTitle className="text-red-400 font-bold">Error</AlertTitle>
              <AlertDescription className="text-red-300">{error}</AlertDescription>
            </Alert>
          )}

        </div>
      </div>
    </div>
  );
}